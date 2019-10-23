import logging
from typing import Tuple

import numpy as np
logger = logging.getLogger('deepspeech')


def mask_features(features, F: int = None, mf: int = None, Tmin: int = 0, Tmax: int = None,
                  mt: int = None, ratio_t: float = None, Tspace: int = 5):
    """ SpecAugment: A Simple Data Augmentation Method. """
    time, channels = features.shape
    means = features.mean(axis=0)       # The mean should be zero if features are normalized
    if F and mf:
        features = mask_frequencies(features, means, channels, F, mf)
    if Tmax and mt:
        features = mask_time(features, means, time, [Tmin, Tmax], mt)
    elif Tmax and ratio_t:                 # Time dimension is chainging so ratio is more appropraite
        features = mask_time_stripes(features, means, time, [Tmin, Tmax], ratio_t, Tspace)
    return features


def mask_frequencies(features, means: np.ndarray, channels: int, F: int, mf: int):
    for i in range(mf):
        f = np.random.random_integers(low=0, high=F)
        f0 = np.random.random_integers(low=0, high=channels-F)
        features[:, f0:f0+f] = means[f0:f0+f]
    return features


def mask_time(features, means: np.ndarray, time: int, T_range: Tuple[int, int], mt: int):
    Tmin, Tmax = T_range
    for i in range(mt):
        t = np.random.random_integers(low=Tmin, high=Tmax)
        t0 = np.random.random_integers(low=0, high=time-Tmax)
        features[t0:t0+t, :] = means
    return features


def mask_time_stripes(features, means: np.ndarray, time: int, T_range: Tuple[int, int], ratio: float,
                      space: int, max_tries: int = 10000):
    booked, masked = np.zeros(time, dtype=bool), 0
    for i in range(max_tries):                                  # Avoid infinite loops (big space value)
        Tmin, Tmax = T_range
        t = np.random.random_integers(low=Tmin, high=Tmax)
        x = np.random.random_integers(low=0, high=time-Tmax)    # This is a start point for masking

        left, right = max(x-space, 0), min(x+t+space, time-1)   # We operate on indices
        if booked[left] or booked[right]:
            continue

        features[x:x+t, :] = means
        booked[left:right] = True
        masked += t

        is_fully_masked = masked >= time*ratio
        if is_fully_masked:
            break
    else:
        logging.warn(f'The partial masked achieved ({masked/time*ratio:.2f})')
    return features
