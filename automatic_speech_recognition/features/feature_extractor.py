import abc
from typing import List
import numpy as np


class FeaturesExtractor:

    def __init__(self, standardize="per_feature"):
        self.standardize = standardize

    def __call__(self,
                 batch_audio: List[np.ndarray]):
        """ Extract features from the file list. """
        features = [self.make_features(audio) for audio in batch_audio]
        lengths = np.array([len(feature) for feature in features])

        X = self.align(features).astype(np.float32)
        if self.standardize:
            X = self.standardize_batch(X, lengths, how=self.standardize)
        return X, lengths

    @abc.abstractmethod
    def make_features(self, audio: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def align(arrays: list, default: int = 0) -> np.ndarray:
        """ Pad arrays (default along time dimensions). Return the single
        array (batch_size, time, features). """
        max_array = max(arrays, key=len)
        X = np.full(shape=[len(arrays), *max_array.shape],
                    fill_value=default, dtype=float)
        for index, array in enumerate(arrays):
            time_dim, features_dim = array.shape
            X[index, :time_dim] = array
        return X

    @staticmethod
    def standardize_batch(features: np.ndarray, seq_lengths: np.array,
                          how="all_features", eps=1e-5) -> np.ndarray:
        """
        Transforms features batch to zero mean and unit std
        in a way specified by how keyword.
        :param features: features have shape (batch_size, time, features)
        :param seq_lengths: length of each feature, (batch_size)
        :param how: {"all_features", "per_feature"}
                   "all_features" - standardize along time * n_features
                   "per_feature" - standardize along time
        :param eps: small constant used to avoid division by zero in stds
        """
        if how == "per_feature":
            means = np.zeros((seq_lengths.shape[0], features.shape[2]))
            stds = np.zeros((seq_lengths.shape[0], features.shape[2]))
            for i in range(features.shape[0]):
                means[i, :] = features[
                    i, : seq_lengths[i], :].mean(axis=0)
                stds[i, :] = features[i, : seq_lengths[i], :].std(axis=0)
            # make sure stds is not zero
            stds += eps
            return (features - np.expand_dims(means, 1)
                    ) / np.expand_dims(stds, 1)
        elif how == "all_features":
            means = np.zeros(seq_lengths.shape[0])
            stds = np.zeros(seq_lengths.shape[0])
            for i in range(features.shape[0]):
                means[i] = features[i, : seq_lengths[i], :].mean()
                stds[i] = features[i, : seq_lengths[i], :].std()
            # make sure stds is not zero
            stds += eps
            return (features - means.reshape((-1, 1, 1))
                    ) / stds.reshape((-1, 1, 1))
        else:
            raise ValueError(f"Unknown how value: {how}")
