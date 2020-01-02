import math
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
                 is_standardization=True,
                 pad_audio_to: int = 0):
        self.features_num = features_num
        self.winfunc = winfunc
        self.frame_len = int(winlen * samplerate)
        self.frame_step = int(winstep * samplerate)
        self.is_standardization = is_standardization
        self.pad_to = pad_audio_to

    def make_features(self, audio: np.ndarray) -> np.ndarray:
        """ Use `python_speech_features` lib to extract log-spectrogram's. """
        audio = self.normalize(audio.astype(np.float32))
        audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
        audio = self.pad(audio) if self.pad_to else audio
        frames = python_speech_features.sigproc.framesig(
            audio, self.frame_len, self.frame_step, self.winfunc
        )
        features = python_speech_features.sigproc.logpowspec(
            frames, self.frame_len, norm=1
        )
        features = features[:, :self.features_num]  # Cut high frequency part
        return self.standardize(features) if self.is_standardization else features

    def pad(self, audio: np.ndarray) -> np.ndarray:
        """ Padding signal is required if you play with mixed precession. """
        length = 1 + int((len(audio) - self.frame_len) // self.frame_step + 1)
        pad_size = (self.pad_to - length % self.pad_to) * self.frame_step
        return np.pad(audio, (0, pad_size), mode='constant')
