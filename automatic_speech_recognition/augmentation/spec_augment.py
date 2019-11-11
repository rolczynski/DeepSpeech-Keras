from typing import Tuple
import numpy as np


class SpecAugment:

    def __init__(self,
                 F: int = None,
                 mf: int = None,
                 Tmin: int = None,
                 Tmax: int = None,
                 mt: int = None):
        """ SpecAugment: A Simple Data Augmentation Method. """
        self.F = F
        self.mf = mf
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.mt = mt

    def __call__(self, batch_features: np.ndarray) -> np.ndarray:
        return np.stack([self.mask_features(features) for features in batch_features], axis=0)

    def mask_features(self, features: np.ndarray) -> np.ndarray:
        features = features.copy()
        time, channels = features.shape
        means = features.mean(axis=0)       # The mean should be zero if features are normalized
        if self.F and self.mf:
            features = self.mask_frequencies(features, means, channels, self.F, self.mf)
        if self.Tmax and self.mt:
            features = self.mask_time(features, means, time, (self.Tmin, self.Tmax), self.mt)
        return features

    @staticmethod
    def mask_frequencies(features: np.ndarray, means: np.ndarray, channels: int, F: int, mf: int):
        for i in range(mf):
            f = np.random.random_integers(low=0, high=F)
            f0 = np.random.random_integers(low=0, high=channels-F)
            features[:, f0:f0+f] = means[f0:f0+f]
        return features

    @staticmethod
    def mask_time(features: np.ndarray, means: np.ndarray, time: int, T_range: Tuple[int, int], mt: int):
        Tmin, Tmax = T_range
        for i in range(mt):
            t = np.random.random_integers(low=Tmin, high=Tmax)
            t0 = np.random.random_integers(low=0, high=time-Tmax)
            features[t0:t0+t, :] = means
        return features
