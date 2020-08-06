import tensorflow as tf
import numpy as np

import librosa
import librosa.filters


if __name__ == "__main__":
    try:
        tf.enable_eager_execution()
    except ValueError as e:
        if e.args[0] != 'tf.enable_eager_execution must be called at program startup.':
            raise e

# persistent variables
_mel_basis = None
_mel_basis_inv = None

def _build_mel_basis(num_mels, sample_rate, num_freq, fmax=None):
    global _mel_basis 
    global _mel_basis_inv
    n_fft = (num_freq - 1) * 2
    M_np = librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels, fmax=fmax)
    M_inv_np = np.linalg.pinv(M_np)
    if np.any(np.isnan(M_inv_np)) or np.any(np.isinf(M_inv_np)):
        raise ValueError('There is a problem with mel inverse')
    _mel_basis = tf.constant(M_np.T, dtype=tf.float32, name="mel_basis")
    _mel_basis_inv = tf.constant(M_inv_np.T, dtype=tf.float32, name="mel_basis_inv")

def _linear_to_mel(spectrogram, num_mels, sample_rate, num_freq, fmax=None):
    global _mel_basis
    if _mel_basis is None:
        _build_mel_basis(num_mels, sample_rate, num_freq)
    melspec = tf.matmul(spectrogram, _mel_basis)
    return melspec

def _mel_to_linear(melspec, num_mels, sample_rate, num_freq, fmax=None):
    """ Reconstruct linear scale spectrogram from mel spectrogram

    Args:
        melspec : linear magnitude mel spectrum, shape=(..., num_mels)
        num_mels : number of mel bins
        sample_rate : audio sample rate
        num_freq : number of frequency bins (one sided)
        fmax : maximum frequency in mel filterbank

    """
    global _mel_basis_inv
    if _mel_basis_inv is None:
        _build_mel_basis(num_mels, sample_rate, num_freq)  
    spectrogram = tf.matmul(melspec, _mel_basis_inv)
    return spectrogram

def _amp_to_db(x, use_db=True):
    if use_db:
        return 20 * tf.math.log(tf.maximum(1e-5, x)) / tf.math.log(10.0)
    else:
        return tf.math.log(tf.maximum(1e-5, x)) / tf.math.log(10.0)

def _db_to_amp(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def _normalize(S, min_level_db=-100, clip=False):
    if clip:
        return tf.clip_by_value((S - min_level_db) / -min_level_db, 0, 1)
    else:
        return (S - min_level_db) / -min_level_db

def _denormalize(S, min_level_db=-100, clip=False):
    if clip:
        return (tf.clip_by_value(S, 0, 1) * -min_level_db) + min_level_db
    else:
        return (S * -min_level_db) + min_level_db

def melspectrogram(X, num_mels, sample_rate, num_freq, ref_level_db=20, clip=False, fmax=None):
    """ Tensorflow mel spectrogram
    Args:
        X : stft magnitudes, shape=(..., num_freq)
        num_mels : number of mel bins
        sample_rate : audio sample rate
        num_freq : number of frequency bins (one sided)
        ref_level_db : used for normalization
        clip : if true, output mel are clipped between [0,1]
        fmax : maximum frequency in mel filterbank

    Returns:
        log mel spectrogram    

    """
    # mel filterbank
    MS = _linear_to_mel(X, num_mels, sample_rate, num_freq, fmax)
    # amp to db
    MS = _amp_to_db(MS) - ref_level_db
    # normalize (clipping optional)
    return _normalize(MS, clip=clip)

def inv_melspectrogram(MS, num_mels, sample_rate, num_freq, ref_level_db=20, clip=False, fmax=None):
    """ Tensorflow inverse mel spectrogram
    
    Reconstructs a magnitude spectrogram using mel pseudoinverse

    Args:
        MS : Mel spectrogram, shape=(..., num_freq)
        num_mels : number of mel bins
        sample_rate : audio sample rate
        num_freq : number of frequency bins (one sided)
        ref_level_db : used for normalization
        fmax : maximum frequency in mel filterbank

    Returns:
        log mel spectrogram    

    """
    
    MS = _denormalize(MS, clip=clip) + ref_level_db
    MS = _db_to_amp(MS)
    S = _mel_to_linear(MS, num_mels, sample_rate, num_freq, fmax)
    S = tf.maximum(S, 1e-6) # clip values smaller than eps
    return S        
