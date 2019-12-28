import numpy as np
import python_speech_features
from .. import features


class FilterBanks(features.FeaturesExtractor):

    def __init__(self, features_num: int, is_standardization=True, **kwargs):
        self.features_num = features_num
        self.is_standardization = is_standardization
        self.params = kwargs

    def make_features(self, audio: np.ndarray) -> np.ndarray:
        """ Use `python_speech_features` lib to extract log filter banks from
        the features file. """
        feat, energy = python_speech_features.fbank(
            audio, nfilt=self.features_num, **self.params
        )
        features = np.log(feat)
        return self.standardize(features) if self.is_standardization else features
