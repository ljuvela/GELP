import tensorflow as tf
import numpy as np
   
import os
import argparse
import json
import shutil

import soundfile as sf   

from model_causal import WaveNet as Generator
from model import WaveNet as Encoder
from model_discriminator import Wavenet as Discriminator

from sigproc import levinson, spec_to_ar
from sigproc import pre_emphasis
from sigproc import ar_analysis_filter, ar_synthesis_filter
from melspec import melspectrogram, inv_melspectrogram

from tensorflow.signal import stft, inverse_stft

from utils import get_stft_loss, get_perceptual_stft_loss
from utils import ganLossContainer
from utils import upsample_cond

from data_provider import NumpyDataProvider


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="./config/config.json")
    parser.add_argument("--initial_model", type=str,
                        default="./sessions/42/model/model")
    parser.add_argument("--suppress_warnings", action='store_true')
    args = parser.parse_args()

    # suppress tensorflow warnings related to 1.X to 2.Y transition 
    if args.suppress_warnings:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    return args


def get_config(filename):
    with open(filename, 'r') as f:
        cfg = json.load(f) 
    return cfg  


def train_model(cfg, config_file):

    args = get_args()
    cfg = get_config(args.config)

    run_id = cfg['run_id']

    batch_size = 1

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

    mel_max_freq = cfg['data'].get('mel_max_freq')
    mel_clip = True

    cond_dim = num_mels
    audio_dim = cfg['data']['audio_dim']

    # data provider
    train_list = os.path.join(cfg['data']['data_dir'], cfg['data']['train_list'])
    audio_dir = os.path.join(cfg['data']['data_dir'], cfg['data']['audio_dir'])

    min_samples = cfg['training']['min_samples']
    max_samples = cfg['training']['max_samples']

    dataprovider = NumpyDataProvider(train_list,
                                     audio_dir,
                                     cfg['data']['audio_ext'],
                                     cfg['data']['audio_dim'],
                                     min_samples=min_samples,
                                     max_samples=max_samples)
        
    x = tf.placeholder(shape=(None,), dtype=tf.float32)    
    x_real_flat = x
    
    # apply pre-emphasis for spectral analysis
    x_emph = pre_emphasis(x, coef=pre_emph_coef)
    # compute short time Fourier transform
    X = stft(x_emph, frame_length, frame_step, fft_length=nfft)
    # get mel spectrogram
    MS = melspectrogram(tf.abs(X), num_mels, sample_rate, num_freq, clip=mel_clip, fmax=mel_max_freq)
    # pseudo invert mel spectrogram 
    X_rec = inv_melspectrogram(MS, num_mels, sample_rate, num_freq, clip=mel_clip, fmax=mel_max_freq)    
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

    # upsample conditioning
    c_upsampled = upsample_cond(x, c, frame_length, frame_step)

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
                          causal=cfg['model']['causal'],
                          add_noise_at_each_layer=cfg['model']['add_noise_at_each_layer']
                          )

    discriminator = Discriminator(name='discriminator',
                                    input_channels=audio_dim,
                                    output_channels=1,
                                    cond_channels=residual_channels,
                                    cond_dim=residual_channels,
                                    dilations=cfg['discriminator']['dilations'],
                                    residual_channels=cfg['discriminator']['residual_channels'],
                                    filter_width=cfg['discriminator']['filter_width'],
                                    postnet_channels=cfg['discriminator']['postnet_channels']
                                    )

    # inverse filter to get 'real' residual excitation 
    exc_real_flat = ar_analysis_filter(x, a, frame_length, frame_step, nfft)
    exc_real = tf.reshape(exc_real_flat, [1, -1, 1])

    # generate excitation
    z = tf.random_normal(shape=tf.shape(exc_real), dtype=tf.float32)
    exc_fake = generator.forward_pass(z, c_upsampled)
    exc_fake_flat = exc_fake[0,:,0]

    # synthesis filter for generated excitation
    x_fake_flat = ar_synthesis_filter(exc_fake_flat, a, frame_length, frame_step, nfft)

    stft_loss_domain = cfg['training']['gan_loss_domain']
    stft_norm_type = cfg['training'].get('stft_norm_type', 2) # default value=2
    stft_loss_multi_window = cfg['training'].get('stft_loss_multi_window', False)
    if (stft_loss_domain == 'glot'):
        stft_loss = get_stft_loss(exc_real_flat, exc_fake_flat,
                                  frame_length, frame_step, nfft,
                                  use_decibels=cfg['training']['stft_loss_in_db'],
                                  norm_type=stft_norm_type)

    elif (stft_loss_domain == 'wave'):
        if cfg['training']['stft_perceptual_loss']:
            stft_loss = get_perceptual_stft_loss(x_real_flat, x_fake_flat, a,
                                      frame_length, frame_step, nfft,
                                      use_decibels=cfg['training']['stft_loss_in_db'],
                                      cancel_pre_emphasis=cfg['training']['stft_loss_cancel_pre_emph'],
                                      pre_emph_coef=pre_emph_coef,
                                      norm_type=stft_norm_type)
        elif stft_loss_multi_window:
            stft_loss = 0.0
            stft_loss += get_stft_loss(
                x_real_flat, x_fake_flat,
                frame_length=128, frame_step=64, nfft=128,
                use_decibels=cfg['training']['stft_loss_in_db'],
                norm_type=stft_norm_type)
            stft_loss += get_stft_loss(
                x_real_flat, x_fake_flat,
                frame_length=512, frame_step=128, nfft=512,
                use_decibels=cfg['training']['stft_loss_in_db'],
                norm_type=stft_norm_type)
            stft_loss += get_stft_loss(
                x_real_flat, x_fake_flat,
                frame_length=2048, frame_step=512, nfft=2048,
                use_decibels=cfg['training']['stft_loss_in_db'],
                norm_type=stft_norm_type)
        else:
            stft_loss = get_stft_loss(x_real_flat, x_fake_flat,
                                      frame_length, frame_step, nfft,
                                      use_decibels=cfg['training']['stft_loss_in_db'],
                                      norm_type=stft_norm_type)

    # image-like tensors for discriminator input
    x_real = tf.reshape(x_real_flat, [1, -1, 1])
    x_fake = tf.reshape(x_fake_flat, [1, -1, 1])

    gan_loss_domain = cfg['training']['gan_loss_domain']
    if (gan_loss_domain == 'glot'):
        gan_losses = ganLossContainer(exc_real,
                                      exc_fake,
                                      cond=c_upsampled,
                                      discriminator=discriminator,
                                      batch_size=32)
    elif (gan_loss_domain == 'wave'):
        gan_losses = ganLossContainer(x_real,
                                      x_fake,
                                      cond=c_upsampled,
                                      discriminator=discriminator,
                                      batch_size=32)
    else:
        raise ValueError(
            'Invalid GAN loss domain "{}", valid options are "glot" and "wave"'.format(gan_loss_domain))
    
    D_loss_gan, G_loss_gan = gan_losses.get_gan_losses(cfg['training']['gan_loss_type'])

    # Generator losses and their tensorboard summaries
    G_loss_total = 0.0
    G_summaries = []
    if cfg['training']['stft_loss_weight'] > 0.0:
        G_loss_total += stft_loss * cfg['training']['stft_loss_weight']
        G_summaries.append(tf.summary.scalar("STFT_loss", stft_loss))
    if cfg['training']['gan_loss_weight'] > 0.0:
        G_loss_total += G_loss_gan * cfg['training']['gan_loss_weight'] 
        G_summaries.append(tf.summary.scalar("G_gan_loss", G_loss_gan))    
    G_summaries.append(tf.summary.scalar("G_total_loss", G_loss_total))
    G_summary_op = tf.summary.merge(G_summaries)

    # Discriminator losses and their tensorboard summaries
    D_loss_total = 0.0
    use_discriminator = False
    D_summaries = []
    if cfg['training']['gan_loss_weight'] > 0.0:
        D_loss_total += D_loss_gan * cfg['training']['gan_loss_weight']
        D_summaries.append(tf.summary.scalar("D_gan_loss", D_loss_gan))
        use_discriminator = True
    if cfg['training']['GP_loss_weight'] > 0.0:
        GP = gan_losses.get_wasserstein_gradient_penalty()
        D_loss_total += GP * cfg['training']['GP_loss_weight']
        D_summaries.append(tf.summary.scalar("Wasserstein_GP", GP))
        use_discriminator = True
    if cfg['training']['R1_loss_weight'] > 0.0:
        R1 = gan_losses.get_R1_gradient_penalty()
        D_loss_total += R1 * cfg['training']['R1_loss_weight']
        D_summaries.append(tf.summary.scalar("R1_GP", R1))
        use_discriminator = True
    D_summaries.append(tf.summary.scalar("D_total_loss", D_loss_total))
    D_summary_op = tf.summary.merge(D_summaries)    

    # create directories for model training and summaries
    run_root = './sessions/{}'.format(run_id)
    for d in ['config', 'tensorboard', 'model', 'audio_snapshots']:
        path = os.path.join(run_root, d)
        if not os.path.isdir(path): 
            os.makedirs(path) 

    model_path = os.path.join(run_root, 'model')
    logs_path =  os.path.join(run_root, 'tensorboard')
    fig_path = os.path.join(run_root, 'audio_snapshots')

    shutil.copy(config_file, os.path.join(run_root, 'config/'))    

    # optimizer settings
    adam_lr = cfg['training']['adam_lr']
    adam_beta1 = cfg['training']['adam_beta1']
    adam_beta2 = cfg['training']['adam_beta2']

    # model variable lists
    theta_e = encoder.get_variable_list()
    theta_g = generator.get_variable_list()
    theta_d = discriminator.get_variable_list()

    # optimizer ops
    G_solver = tf.train.AdamOptimizer(
        learning_rate=adam_lr,
        beta1=adam_beta1,
        beta2=adam_beta2).minimize(G_loss_total, var_list=[theta_g, theta_e])

    if use_discriminator:
        D_solver = tf.train.AdamOptimizer(
            learning_rate=adam_lr,
            beta1=adam_beta1,
            beta2=adam_beta2).minimize(D_loss_total, var_list=theta_d)

    # restore training status (optional)
    training_status_file = os.path.join(model_path, "status.npz")
    restore = os.path.isfile(training_status_file)
    if restore:
        status = np.load(training_status_file)
        iter_total = int(status['iter_total']) + 1
        epoch_start = int(status['epoch']) + 1
    else:
        iter_total = 0
        epoch_start = 0    

    with tf.Session() as sess:

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        saver = tf.train.Saver()
        if restore:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))

        iter_ind = 0
        for epoch in range(epoch_start, cfg['training']['num_epochs']):
            for x_np in dataprovider:
                # cut to a multiple of frame step
                n_samples = frame_step * (x_np.shape[0] // frame_step)
                x_np = x_np[:n_samples]
            
                if use_discriminator:
                    _, D_loss_np, summary_D = sess.run([D_solver, D_loss_total, D_summary_op],
                                                        feed_dict={x: x_np})
                    if not np.all(np.isfinite(D_loss_np)):
                        raise ValueError(
                            "Discriminator loss is not finite, stopping training")
                    if iter_ind % cfg['training']['summary_interval'] == 0:                        
                        summary_writer.add_summary(summary_D, iter_total)

                _, G_loss_np, summary_G = sess.run([G_solver, G_loss_total, G_summary_op],
                                                      feed_dict={x: x_np})
                if not np.all(np.isfinite(G_loss_np)):
                    raise ValueError(
                        "Generator loss is not finite, stopping training")
                if iter_ind % cfg['training']['summary_interval'] == 0:                         
                    summary_writer.add_summary(summary_G, iter_total)

                # audio checkpoint
                if iter_ind % cfg['training']['audio_snapshot_interval'] == 0:                    
                    x_real_np, x_fake_np, exc_real_np, exc_fake_np = sess.run([x_real_flat, x_fake_flat, exc_real_flat, exc_fake_flat],
                                                                   feed_dict={x: x_np})
                    sf.write(os.path.join(fig_path, 'iter{}-target.wav'.format(iter_total)),
                             x_real_np, sample_rate)
                    sf.write(os.path.join(fig_path, 'iter{}-generated.wav'.format(iter_total)),
                             x_fake_np, sample_rate)
                    sf.write(os.path.join(fig_path, 'iter{}-exc_target.wav'.format(iter_total)),
                             exc_real_np, sample_rate)   
                    sf.write(os.path.join(fig_path, 'iter{}-exc_generated.wav'.format(iter_total)),
                             exc_fake_np, sample_rate)                                

                # save model
                save_interval = cfg['training']['model_save_interval']
                if iter_ind % save_interval == 0:
                    saver.save(sess, os.path.join(model_path, "model_iter{}.ckpt".format(iter_total)))
                    np.savez(training_status_file, iter_total=iter_total, epoch=epoch)

                iter_ind += 1
                iter_total += 1

                if iter_ind > cfg['training']['max_iters']:
                    break

if __name__ == "__main__":

    args = get_args()
    cfg = get_config(args.config)

    train_model(cfg, config_file=args.config)
