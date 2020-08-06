import numpy as np

import scipy

import tensorflow as tf
from tensorflow.signal import stft, inverse_stft

if __name__ == "__main__":
    try:
        tf.enable_eager_execution()
    except ValueError as e:
        if e.args[0] != 'tf.enable_eager_execution must be called at program startup.':
            raise e

def pre_emphasis(x, coef=0.97):
    y = x[1:] - coef * x[:-1]
    return tf.concat([x[0:1], y], axis=0)

def spec2poly_ref(spec, p):
    import scipy.linalg as LA
    a_list = []
    for i, spec_vec in enumerate(spec.T):

        # floor reconstructed spectrum
        spec_vec = np.maximum(spec_vec, 1e-9)

        # squared magnitude 2-sided spectrum
        twoside = np.concatenate([spec_vec, np.flipud(spec_vec[1:-1])])
        twoside = np.square(twoside) 
        r = np.fft.ifft(twoside)
        r = r.real

        # levinson-durbin
        a = LA.solve_toeplitz(r[0:p],r[1:p+1])
        a = np.r_[1.0, -1.0*a]

        a_list.append(a)

    return np.stack(a_list).T

def forward_levinson(gamma, M):
    # gamma is reflection coefficients, gamma.shape == (M, N)
    # M is filter order (int)
    #     
    l = tf.concat([tf.ones_like(gamma[0:1]), tf.zeros_like(gamma[0:M])], axis=0)
    l_prev = l 
    for p in range(0, M):
        pad = tf.maximum(M-p-1, 0)
        if p == 0:
            l = tf.concat([-1.0*gamma[p:p+1],
                           tf.ones_like(gamma[0:1]),
                           tf.zeros_like(gamma[0:pad])], axis=0)
        else:
            l = tf.concat([-1.0*gamma[p:p+1],
                           l_prev[0:p] - 1.0*gamma[p:p+1] * tf.reverse(l_prev[0:p], axis=[0]), # should be complex conjugate, if complex vals are used
                           tf.ones_like(gamma[0:1]),
                           tf.zeros_like(gamma[0:pad])], axis=0)
        l_prev = l

    l = tf.reverse(l, axis=[0]) # flip zero delay to zero:th index
    return l


def levinson(R, M):

    E = R[0:1]

    L = tf.concat([tf.ones_like(R[0:1]), tf.zeros_like(R[0:M])], axis=0)
    L_prev = L
    for p in range(0, M):
        gamma = tf.reduce_sum(L_prev[0:p+1] * R[1:p+2], axis=0) / E
        pad = np.maximum(M-p-1, 0)
        if p == 0:
            L = tf.concat([-1.0*gamma,
                           tf.ones_like(R[0:1]),
                           tf.zeros_like(R[0:pad])], axis=0)
        else:
            L = tf.concat([-1.0*gamma,
                           L_prev[0:p] - 1.0*gamma *
                           tf.reverse(L_prev[0:p], axis=[0]),
                           tf.ones_like(R[0:1]),
                           tf.zeros_like(R[0:pad])], axis=0)
        L_prev = L

        E = E * (1.0 - tf.square(gamma))  # % order-p mean-square error

    L = tf.reverse(L, axis=[0])  # flip zero delay to zero:th index
    return L


def levinson_inner(R, M):
    # same as levinson, but operate on rightmost tensor dimension (axis=-1)

    E = R[..., 0:1]

    L = tf.concat([tf.ones_like(R[..., 0:1]),
                   tf.zeros_like(R[..., 0:M])], axis=-1)
    L_prev = L
    for p in range(0, M):
        gamma = tf.reduce_sum(L_prev[..., 0:p+1] * R[..., 1:p+2], axis=-1, keepdims=True) / E
        pad = np.maximum(M-p-1, 0)
        if p == 0:
            L = tf.concat([-1.0*gamma,
                           tf.ones_like(R[..., 0:1]),
                           tf.zeros_like(R[..., 0:pad])], axis=-1)
        else:
            L = tf.concat([-1.0*gamma,
                           L_prev[..., 0:p] - 1.0*gamma *
                           tf.reverse(L_prev[..., 0:p], axis=[-1]),
                           tf.ones_like(R[..., 0:1]),
                           tf.zeros_like(R[..., 0:pad])], axis=-1)
        L_prev = L

        E = E * (1.0 - tf.square(gamma))  # % order-p mean-square error

    L = tf.reverse(L, axis=[-1])  # flip zero delay to zero:th index
    return L


def spec_to_ar(X, filter_order):
    x = tf.square(tf.abs(X))  # squared magnitude spectrum
    x = tf.maximum(x, 1e-9)
    twoside = tf.concat([x, tf.reverse(x[..., 1:-1], axis=[-1])], axis=-1)
    twoside = tf.cast(twoside, tf.complex64)
    r = tf.signal.ifft(twoside)
    r = tf.math.real(r)
    r = tf.transpose(r)
    a = levinson(r, filter_order)
    a = tf.transpose(a)
    return a

def window_fn_cosine(window_length, dtype=tf.float32):
    win = tf.signal.hann_window(window_length, periodic=True, dtype=dtype)
    return tf.sqrt(win)

def energy_from_spectrogram(X):
    """ Calculate frame energies from magnitude or complex spectrogram """
    energy = tf.maximum(tf.reduce_sum(tf.abs(X[..., 1:]) ** 2.0, axis=-1, keepdims=True), 1e-6)
    return energy

def ar_analysis_filter(x, a, frame_length, frame_step, nfft, energy=None):

    a_pad = tf.pad(a, paddings=[[0, 0], [0, nfft - tf.shape(a)[-1]]])
    A = tf.signal.rfft(a_pad)

    # cosine windows for fft analysis and synthesis
    X = stft(x, frame_length, frame_step,
             fft_length=nfft, window_fn=window_fn_cosine)
    E = A * X

    if energy is not None:
        E_energy = energy_from_spectrogram(E)
        E *= tf.cast(tf.sqrt(energy / E_energy), tf.complex64)

    # prediction residual in time domain
    e = inverse_stft(E, frame_length, frame_step,
                     fft_length=nfft, window_fn=window_fn_cosine)

    gain = 2 * frame_step / frame_length
    return gain * e


def ar_synthesis_filter(x, a, frame_length, frame_step, nfft, energy=None):

    a_pad = tf.pad(a, paddings=[[0, 0], [0, nfft - tf.shape(a)[-1]]])
    A = tf.signal.rfft(a_pad)
 
    E = stft(x, frame_length, frame_step,
             fft_length=nfft, window_fn=window_fn_cosine)

    # synthesis filter
    amp = 1.0 / tf.maximum(tf.abs(A), 1e-9)
    phase = -1.0 * tf.angle(A)
    H = tf.cast(amp, tf.complex64) * tf.exp(1j * tf.cast(phase, tf.complex64))

    Y = H * E

    if energy is not None:
        Y_energy = energy_from_spectrogram(Y)
        Y *= tf.cast(tf.sqrt(energy / Y_energy), tf.complex64)

    y = inverse_stft(Y, frame_length, frame_step,
                     fft_length=nfft, window_fn=window_fn_cosine)

    gain = 2 * frame_step / frame_length
    return gain * y
