import numpy as np
import python_speech_features
from .. import features


class Spectrogram(features.FeaturesExtractor):

    def __init__(self,
                 features_num: int,
                 samplerate: int,
                 winlen: float,
                 winstep: float,
                 winfunc=None,
                 is_standardization=True):
        self.features_num = features_num
        self.winfunc = winfunc
        self.frame_len = int(winlen * samplerate)
        self.frame_step = int(winstep * samplerate)
        self.is_standardization = is_standardization

    def make_features(self, audio: np.ndarray) -> np.ndarray:
        """ Use `python_speech_features` lib to extract log-spectrogram's. """
        frames = python_speech_features.sigproc.framesig(
            audio, self.frame_len, self.frame_step, self.winfunc
        )
        features = python_speech_features.sigproc.logpowspec(
            frames, self.frame_len, norm=1
        )
        features = features[:, :self.features_num]  # Cut high frequency part
        return self.standardize(features) if self.is_standardization else features
