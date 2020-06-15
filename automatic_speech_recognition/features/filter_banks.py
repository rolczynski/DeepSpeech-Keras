import numpy as np
import math
import librosa
from .. import features
from . import audio_utils


class FilterBanks(features.FeaturesExtractor):

    def __init__(self, features_num: int, sample_rate: int = 16000,
                 winlen: float = 0.02, winstep: float = 0.01,
                 window="hann", n_fft=None, mag_power=2.0,
                 log=True, log_eps: float = 2**(-24),
                 dither: float = 1e-5, preemph: float = 0.97,
                 standardize="per_feature"):

        self.features_num = features_num
        self.sample_rate = sample_rate
        self.win_length = math.ceil(winlen * sample_rate)
        self.hop_length = math.ceil(winstep * sample_rate)
        self.window = window
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.mag_power = mag_power
        self.log = log
        self.log_eps = log_eps
        self.dither = dither
        self.preemph = preemph
        super().__init__(standardize=standardize)

    def make_features(self, audio: np.ndarray) -> np.ndarray:
        """ Extract (log) mel filter banks from the audio """
        # dither
        if self.dither > 0:
            audio = audio_utils.dither(audio, self.dither)

        # do preemphasis
        if self.preemph is not None:
            audio = audio_utils.preemphasize(audio, self.preemph)

        # get filterbanks. Librosa returns (n_mels, t)
        features = librosa.feature.melspectrogram(
            audio, self.sample_rate, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, center=True, power=self.mag_power,
            n_mels=self.features_num
        )

        # logarithm if needed
        if self.log:
            features = np.log(features + self.log_eps)

        # put features into correct order (time, n_features)
        features = features.transpose()
            
        return features
