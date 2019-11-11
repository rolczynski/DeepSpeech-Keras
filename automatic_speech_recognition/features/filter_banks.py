import numpy as np
import python_speech_features
from .. import features


class FilterBanks(features.FeaturesExtractor):

    def __init__(self, **kwargs):
        if 'winfunc' in kwargs and kwargs['winfunc'] == 'hamming':
            kwargs['winfunc'] = np.hamming
        self.params = kwargs

    def make_features(self, audio: np.ndarray) -> np.ndarray:
        """ Use `python_speech_features` lib to extract features from the features file. """
        feat, energy = python_speech_features.fbank(audio, **self.params)
        features = np.log(feat)
        return features
