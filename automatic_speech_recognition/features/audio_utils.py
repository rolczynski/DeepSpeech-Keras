"""Common operations for audio data"""
import numpy as np


def normalize(audio: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """ Normalize float32 signal to [-1, 1] range. """
    gain = 1.0 / (np.max(np.abs(audio)) + eps)
    return audio * gain


def dither(audio: np.ndarray, level: float = 1e-5) -> np.ndarray:
    """ Adds random noise at a specified level"""
    return audio + level * np.random.rand(*audio.shape)


def preemphasize(audio: np.ndarray, alpha: float) -> np.ndarray:
    """ Preemphasizes the signal to improve high frequencies"""
    return np.append(audio[0], audio[1:] - alpha * audio[:-1])
