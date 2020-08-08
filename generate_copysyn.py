import tensorflow as tf

import numpy as np
   
import os
import argparse
import json
import shutil
import soundfile as sf 
from glob import glob    

from model_causal import WaveNet as Generator
from cond_model import UpsampleBilinearInterp
from model import WaveNet as Encoder
from model_discriminator import Wavenet as Discriminator

from tensorflow.signal import stft, inverse_stft
from sigproc import levinson, spec_to_ar
from sigproc import ar_analysis_filter, ar_synthesis_filter
from melspec import melspectrogram, inv_melspectrogram

from utils import get_stft_loss, get_perceptual_stft_loss
from utils import ganLossContainer
from utils import noise_gate

from data_provider import NumpyDataProvider

def pre_emphasis(x, coef=0.97):
    y = x[1:] - coef * x[:-1]
    return tf.concat([x[0:1], y], axis=0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str,
                        default='sessions/42')                    
    parser.add_argument('--input_wav_dir', type=str) 
    args = parser.parse_args()
    return args


def get_config(filename):
    with open(filename, 'r') as f:
        cfg = json.load(f) 
    return cfg  


def copy_synthesis(args):

    config = os.path.join(args.model_dir, 'config', 'config.json') 
    cfg = get_config(config)

    run_id = cfg['run_id']

    residual_channels = cfg['model']['residual_channels']
    postnet_channels = cfg['model']['postnet_channels']
    cond_embed_dim = cfg['model']['cond_embed_dim']
    dilations = cfg['model']['dilations']
    filter_width = cfg['model']['filter_width']

    filter_order = cfg['data']['ar_filter_order']
    num_freq = cfg['data']['num_freq']
    nfft = 2 * (num_freq - 1)
    sample_rate = cfg['data']['sample_rate']
    frame_shift_ms = cfg['data']['frame_shift_ms']
    frame_length_ms = cfg['data']['frame_length_ms']
    frame_step = int(frame_shift_ms / 1000 * sample_rate)
    frame_length = int(frame_length_ms / 1000 * sample_rate)
    num_mels = cfg['data']['num_mels']
    pre_emph_coef = cfg['data']['preemphasis']

    cond_dim = num_mels
    audio_dim = cfg['data']['audio_dim']

    upsampling_ratio = frame_step

    # Read .wav files from directory if provided
    if args.input_wav_dir is not None:
        audio_dir = args.input_wav_dir
        files = glob(os.path.join(audio_dir, '*.wav'))
        test_list = [os.path.splitext(os.path.basename(f))[0] for f in files]
        
    else:
        test_list_filename = cfg['data'].get('test_list', None)
        if test_list_filename is None:
            raise ValueError("Test file list not found in config. \n \
                Please modify the config or provide input wave file directory by the '--input_wav_dir' argument"  )
        
        test_list = os.path.join(cfg['data']['data_dir'], test_list_filename)
        audio_dir = os.path.join(cfg['data']['data_dir'], cfg['data']['audio_dir'])


    min_samples = 1
    max_samples = 20 * sample_rate

    dataprovider = NumpyDataProvider(test_list,
                                     audio_dir,
                                     cfg['data']['audio_ext'],
                                     cfg['data']['audio_dim'],
                                     min_samples=min_samples,
                                     max_samples=max_samples,
                                     return_basename=True)
        
    x = tf.placeholder(shape=(None,), dtype=tf.float32)    
    x_real_flat = x
    
    melspec_clip = True

    # apply pre-emphasis for spectral analysis
    x_emph = pre_emphasis(x, coef=pre_emph_coef)
    # compute short time Fourier transform
    X = stft(x_emph, frame_length, frame_step, fft_length=nfft)
    # get mel spectrogram
    MS = melspectrogram(tf.abs(X), num_mels, sample_rate, num_freq, clip=melspec_clip)
    # pseudo invert mel spectrogram 
    X_rec = inv_melspectrogram(MS, num_mels, sample_rate, num_freq, clip=melspec_clip)    
    # fit AR envelope
    a = spec_to_ar(X_rec, filter_order)

    # conditioning model
    ms_img = tf.expand_dims(MS, 0)
    use_condnet_dropout = False
    encoder = Encoder(name='encoder',
                      input_channels=cond_dim,
                      output_channels=residual_channels,
                      dilations=cfg['cond_model']['dilations'],
                      filter_width=cfg['cond_model']['filter_width'],
                      postnet_channels=cfg['cond_model']['postnet_channels'],
                      use_dropout=use_condnet_dropout)
    c = encoder.forward_pass(ms_img)

    # padding for cond
    x_shape = tf.shape(x)
    num_frames = tf.ceil((x_shape[0] - frame_length + frame_step) / frame_step)
    frame_diff = tf.floor(x_shape[0] / frame_step) - num_frames
    c_pad_left = tf.cast(tf.ceil(frame_diff / 2), dtype=tf.int32)
    c_pad_right = tf.cast(tf.floor(frame_diff / 2), dtype=tf.int32)
    c_padded = tf.pad(c, [[0, 0], [c_pad_left, c_pad_right], [0, 0]], 'REFLECT')

    upsampler = UpsampleBilinearInterp(upsample_factor=upsampling_ratio,
                                       channels=residual_channels)
    c_upsampled = upsampler.forward_pass(c_padded)

    # generator model
    generator = Generator(name='generator',
                          input_channels=1,
                          output_channels=1,
                          residual_channels=residual_channels,
                          postnet_channels=postnet_channels,
                          filter_width=filter_width,
                          dilations=dilations,
                          cond_channels=residual_channels,
                          cond_embed_dim=residual_channels,
                          causal=cfg['model'].get('causal', False)
                          )

    # noise excitation
    z = tf.random_normal(shape=tf.shape(c_upsampled[..., 0:1]), dtype=tf.float32)
    # generator output is residual excitation signal
    exc_fake = generator.forward_pass(z, c_upsampled)
    exc_fake_flat = exc_fake[0,:,0]

    # synthesis filter
    x_fake_flat = ar_synthesis_filter(exc_fake_flat, a, frame_length, frame_step, nfft)

    run_root = './sessions/{}'.format(run_id)
    model_path = os.path.join(run_root, 'model')
    syn_path = os.path.join(run_root, 'syn')
    exc_path = os.path.join(run_root, 'exc')

    if not os.path.isdir(syn_path): 
        os.makedirs(syn_path) 
    if not os.path.isdir(exc_path): 
        os.makedirs(exc_path) 

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(model_path, "model.ckpt"))

        for x_np, bname in dataprovider:
            # cut to a multiple of frame step
            n_samples = frame_step * (x_np.shape[0] // frame_step)
            x_np = x_np[:n_samples]

            x_fake_np, exc_fake_np = sess.run([x_fake_flat, exc_fake_flat],
                                               feed_dict={x: x_np})

            print(f'Generating {bname}')
            x_fake_np = noise_gate(x_fake_np, threshold=-40, reduction=-15)                                                            
            sf.write(os.path.join(syn_path, '{}.syn.wav'.format(bname)),
                        x_fake_np, sample_rate)
            sf.write(os.path.join(exc_path, '{}.exc.wav'.format(bname)),
                        exc_fake_np, sample_rate)                                


if __name__ == "__main__":

    args = get_args()
    copy_synthesis(args)
