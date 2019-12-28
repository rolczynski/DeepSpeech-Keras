import abc
from typing import List
import numpy as np


class FeaturesExtractor:

    def __call__(self, batch_audio: List[np.ndarray]) -> np.ndarray:
        """ Extract features from the file list. """
        features = [self.make_features(audio) for audio in batch_audio]
        X = self.align(features)
        return X

    @abc.abstractmethod
    def make_features(self, audio: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def standardize(features: np.ndarray) -> np.ndarray:
        """ Standardize globally, independently of features. """
        mean = np.mean(features)
        std = np.std(features)
        return (features - mean) / std

    @staticmethod
    def align(arrays: list, default=0) -> np.ndarray:
        """ Pad arrays (default along time dimensions). Return the single
        array (batch_size, time, features). """
        max_array = max(arrays, key=len)
        X = np.full(shape=[len(arrays), *max_array.shape],
                    fill_value=default,
                    dtype=np.float64)
        for index, array in enumerate(arrays):
            time_dim, features_dim = array.shape
            X[index, :time_dim] = array
        return X
