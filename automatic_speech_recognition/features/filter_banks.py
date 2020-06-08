import numpy as np
import python_speech_features
from tensorflow.python.ops import gen_audio_ops as contrib_audio
import tensorflow as tf
from .. import features


class FilterBanks(features.FeaturesExtractor):

    def __init__(self, features_num: int, is_standardization=True, **kwargs):
        super().__init__()
        self.features_num = features_num
        self.is_standardization = is_standardization
        self.params = kwargs

        self.features_avg = []

    def make_features(self, audio: np.ndarray) -> np.ndarray:
        """ Use `python_speech_features` lib to extract log filter banks from
        the features file. """
        feat, energy = python_speech_features.fbank(
            audio, nfilt=self.features_num, **self.params
        )
        features = np.log(feat)
        return self.standardize(features) if self.is_standardization else features

    
class MFCC(features.FeaturesExtractor):

    def __init__(self, features_num: int, is_standardization=False, sample_rate=16000, 
    winlen=0.02, winstep=0.01):
        super().__init__()
        self.features_num = features_num
        self.is_standardization = is_standardization
        self.sample_rate = sample_rate
        self.window_size = int(winlen * sample_rate)
        self.window_step = int(winstep * sample_rate)

    def make_features(self, audio: np.ndarray) -> np.ndarray:
        """ Use `python_speech_features` lib to extract log filter banks from
        the features file. """
        spectrogram = contrib_audio.audio_spectrogram(audio.audio,
                                                        window_size=self.window_size,
                                                        stride=self.window_step,
                                                        magnitude_squared=True)
        mfccs = contrib_audio.mfcc(spectrogram=spectrogram,
                                        sample_rate=self.sample_rate,
                                        dct_coefficient_count=self.features_num,
                                        upper_frequency_limit=self.sample_rate//2)
        return self.standardize(mfccs[0]) if self.is_standardization else mfccs[0]
