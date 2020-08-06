import tensorflow as tf
import numpy as np

from tensorflow.contrib.signal import stft

from melspec import _amp_to_db as amp_to_db
from sigproc import pre_emphasis

import soundfile as sf
from scipy.signal import filtfilt
import librosa

from cond_model import UpsampleBilinearInterp


if __name__ == "__main__":
    try:
        tf.enable_eager_execution()
    except ValueError as e:
        if e.args[0] != 'tf.enable_eager_execution must be called at program startup.':
            raise e


def get_perceptual_stft_loss(x_real, x_fake, a, frame_length, frame_step, nfft,
                             use_decibels=False, cancel_pre_emphasis=False, pre_emph_coef=0.97,
                             norm_type=2):

    x_real_stft = stft(x_real, frame_length, frame_step, fft_length=nfft)
    x_fake_stft = stft(x_fake, frame_length, frame_step, fft_length=nfft)

    filter_length = tf.shape(a)[-1]
    alpha = 0.9 
    w = tf.pow(alpha, tf.range(0.0, tf.cast(filter_length, tf.float32)))

    # Original CELP masking filter
    # a_w = w * a
    # a_pad = tf.pad(a, paddings=[[0, 0], [0, nfft - filter_length]])
    # a_w_pad = tf.pad(a_w, paddings=[[0, 0], [0, nfft - filter_length]])
    # A = tf.spectral.rfft(a_pad)
    # A_w = tf.spectral.rfft(a_w_pad)
    # W = tf.abs(A) * tf.abs(A_w)

    # Tom's simplified masking filter
    a_w = w * a
    a_w_pad = tf.pad(a_w, paddings=[[0, 0], [0, nfft - filter_length]])
    A_w = tf.spectral.rfft(a_w_pad)
    W = tf.abs(A_w)

    energy = tf.reduce_mean(tf.square(tf.abs(x_real_stft)) + 1e-2)

    # de-emphasis 
    if cancel_pre_emphasis:
        b = tf.constant([1.0, -1.0 * pre_emph_coef])
        b_pad = tf.pad(b, paddings=[[0, nfft - 2]])
        B = tf.spectral.rfft(b_pad)
        W *= tf.expand_dims(1.0 / tf.abs(B), axis=0)

    if use_decibels:
        stft_loss = tf.reduce_mean(tf.square(
            amp_to_db(W * tf.abs(x_real_stft) / (tf.abs(x_fake_stft) + 1e-5))
        ))
        stft_loss -= amp_to_db(energy)

    else:
        if norm_type == 2:
            stft_loss = tf.reduce_mean(tf.square(
                W * (tf.abs(x_real_stft) - tf.abs(x_fake_stft))
            ))
        elif norm_type == 1:
            stft_loss = tf.reduce_mean(tf.abs(
                W * (tf.abs(x_real_stft) - tf.abs(x_fake_stft))
            ))
        else:
            raise ValueError('invalid norm type {}, valid options are 1 and 2'.format(norm_type))   
        stft_loss /= energy

    return stft_loss  

def get_stft_loss(x_real, x_fake, frame_length, frame_step, nfft, use_decibels=False, norm_type=2):
    
    x_real_stft = stft(x_real, frame_length, frame_step, fft_length=nfft)
    x_fake_stft = stft(x_fake, frame_length, frame_step, fft_length=nfft)

    energy = tf.reduce_mean(tf.square(tf.abs(x_real_stft)) + 1e-2)

    if use_decibels:
        stft_loss = tf.reduce_mean(tf.square(
            amp_to_db(tf.abs(x_real_stft)) - amp_to_db(tf.abs(x_fake_stft))
        ))
        # stft_loss -= amp_to_db(energy)
    else:
        if norm_type == 2:
            stft_loss = tf.reduce_mean(tf.square(
                tf.abs(x_real_stft) - tf.abs(x_fake_stft)
            ))
        elif norm_type == 1:
            stft_loss = tf.reduce_mean(tf.abs(
                tf.abs(x_real_stft) - tf.abs(x_fake_stft)
            ))
        else:
            raise ValueError('invalid norm type {}, valid options are 1 and 2'.format(norm_type))     
        stft_loss /= energy

    return stft_loss                                         

def random_crop_batch(x, batch_size, width):
    shape = (tf.shape(x)[0], width, tf.shape(x)[-1])
    batch_list = []
    for _ in range(batch_size):
        x_crop = tf.random_crop(x, shape)
        batch_list.append(x_crop)
    return tf.concat(batch_list, axis=0)    

def wasserstein_gradient_penalty(x_real, x_fake, cond, discriminator, weight=10.0):
        epsilon_shape = tf.stack([tf.shape(x_real)[0], 1, 1])
        epsilon = tf.random_uniform(epsilon_shape, 0.0, 1.0)
        x_hat = epsilon * x_real + (1.0-epsilon) * x_fake
        d_hat = discriminator.forward_pass(x_hat, cond)
        gradients = tf.gradients(d_hat, x_hat)[0]
        gradnorm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
        gradient_penalty = weight * tf.reduce_mean(tf.square(gradnorm - 1.0))
        return gradient_penalty

def R1_gradient_penalty(x, discriminator_out):
    gradients = tf.gradients(discriminator_out, x)
    R1 = 0.0
    for g in gradients:
        gradnorm = tf.sqrt(tf.reduce_sum(tf.square(g), axis=[1,2]))
        R1 += tf.reduce_mean(tf.square(gradnorm))
    return R1        

def get_gan_losses(x_real, x_fake, c_upsampled, discriminator, batch_size=32):

    R = discriminator.get_receptive_field()
    valid_width = 2*R+1
    tmp = random_crop_batch(tf.concat([x_fake, x_real, c_upsampled], axis=-1),
                            batch_size=batch_size, width=valid_width)
    x_fake_batch = tmp[:, :, 0:1]
    x_real_batch = tmp[:, :, 1:2]
    c_batch = tmp[:, :, 2:]

    # Feed to discriminator
    D_out_real = discriminator.forward_pass(x_real_batch, c_batch)
    D_out_fake = discriminator.forward_pass(x_fake_batch, c_batch)

    # WGAN loss
    D_loss_gan = -tf.reduce_mean(D_out_real) + tf.reduce_mean(D_out_fake) 
    G_loss_gan = -tf.reduce_mean(D_out_fake)

    # R1 gradient penalty 
    R1 = R1_gradient_penalty(x_real_batch, D_out_real)

    # Wasserstein gradient penalty
    GP = wasserstein_gradient_penalty(x_real_batch, x_fake_batch,
                                      c_batch, discriminator, weight=1.0)

    return D_loss_gan, G_loss_gan, GP, R1

def noise_gate(x, frame_len=400, frame_step=80, threshold=-40, reduction=-15, mode='gate', return_gain=False):

    # librosa.util.frame returns a re-strided reference to original input
    # If frames are modified in-place, also the original is,
    # prevent this by passing a copy
    x_framed = librosa.util.frame(x.copy(), frame_length=frame_len, hop_length=80)

    win = np.expand_dims(np.hanning(frame_len), -1)
    x_framed *= win
    energy = np.mean(np.square(x_framed), axis=0)
    gain = 20.0 * np.log10(energy * 1e5) 

    if mode == 'gate':
        reduce_idx = np.where(gain < threshold)
    elif mode == 'compressor':
        reduce_idx = np.where(gain > threshold)
    gain_reduction = np.zeros_like(gain)
    gain_reduction[reduce_idx] += reduction

    # smooth gain reduction
    gain_reduction = filtfilt([1.0], [1.0 , -0.5], gain_reduction)
    gain_reduction *= reduction / (gain_reduction.min() + 1e-6) 

    t_sample = np.linspace(0.0, 1.0, x.shape[0])
    t_frame = np.linspace(0.0, 1.0, energy.shape[0])
    gain_sample = np.interp(t_sample, t_frame, gain_reduction)

    gain_lin = np.power(10.0, gain_sample / 20.0) 

    x_gated = gain_lin * x
    if return_gain:
        return x_gated, gain
    else:    
        return x_gated
class ganLossContainer:

    def __init__(self, x_real, x_fake, cond, discriminator, batch_size=32):
        
        self.x_real = x_real
        self.x_fake = x_fake
        self.cond = cond
        self.discriminator = discriminator
        self.batch_size = batch_size

        R = self.discriminator.get_receptive_field()
        valid_width = 2*R+1
        tmp = random_crop_batch(tf.concat([self.x_fake, self.x_real, self.cond], axis=-1),
                                batch_size=batch_size, width=valid_width)
        self.x_fake_batch = tmp[:, :, 0:1]
        self.x_real_batch = tmp[:, :, 1:2]
        self.c_batch = tmp[:, :, 2:]

        self.D_out_real = discriminator.forward_pass(self.x_real_batch, self.c_batch)
        self.D_out_fake = discriminator.forward_pass(self.x_fake_batch, self.c_batch)

    def get_gan_losses(self, loss_type):
        if loss_type == 'wgan':
            D_loss = -tf.reduce_mean(self.D_out_real) \
                     + tf.reduce_mean(self.D_out_fake)
            G_loss = -tf.reduce_mean(self.D_out_fake)
        elif loss_type == 'lsgan':
            D_loss = tf.reduce_mean(tf.square(self.D_out_real - 1.0)) \
                    + tf.reduce_mean(tf.square(self.D_out_fake))
            G_loss = tf.reduce_mean(tf.square(self.D_out_fake - 1.0))
        elif loss_type == 'gan':
            D_real = tf.sigmoid(self.D_out_real)
            D_fake = tf.sigmoid(self.D_out_fake)
            D_loss = -tf.reduce_mean(tf.log(D_real)) \
                     - tf.reduce_mean(tf.log(1.0 - D_fake))
            G_loss = -tf.reduce_mean(tf.log(D_fake))
        else:
            raise ValueError('Loss type "{}" not supported'.format(loss_type))

        return D_loss, G_loss

    def get_wasserstein_gradient_penalty(self):
        return wasserstein_gradient_penalty(self.x_real_batch, self.x_fake_batch,
                                        self.c_batch, self.discriminator, weight=1.0)
        
    def get_R1_gradient_penalty(self):
        return R1_gradient_penalty(self.x_real_batch, self.D_out_real)

def upsample_cond(x, c, frame_length, frame_step):
    """ Upsampling for conditioning

    Args: 
        x : signal, only shape is used
        c : conditioning
        frame_length : analysis frame length (int)
        frame_step: analysis frame hop size (int)

    Returns:
        c upsampled to match x timesteps     
    """

    upsampling_ratio = frame_step

    # padding for cond
    x_shape = tf.shape(x)
    c_shape = tf.shape(c)
    num_frames = tf.ceil((x_shape[0] - frame_length + frame_step) / frame_step)
    frame_diff = tf.floor(x_shape[0] / frame_step) - num_frames
    c_pad_left = tf.cast(tf.ceil(frame_diff / 2), dtype=tf.int32)
    c_pad_right = tf.cast(tf.floor(frame_diff / 2), dtype=tf.int32)
    c_padded = tf.pad(c, [[0, 0], [c_pad_left, c_pad_right], [0, 0]], 'REFLECT')

    residual_channels = c_shape[-1]
    upsampler = UpsampleBilinearInterp(upsample_factor=upsampling_ratio,
                                       channels=residual_channels)
    c_upsampled = upsampler.forward_pass(c_padded)
    return c_upsampled
