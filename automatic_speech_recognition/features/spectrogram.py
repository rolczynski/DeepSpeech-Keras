import math
import numpy as np
import librosa
from .. import features
from . import audio_utils


class Spectrogram(features.FeaturesExtractor):
    def __init__(self, features_num: int, sample_rate: int = 16000,
                 winlen: float = 0.02, winstep: float = 0.01,
                 window="hann", n_fft=None, mag_power=1.0,
                 log=True, log_eps: float = 2**(-24),
                 dither: float = 1e-5, preemph: float = 0.97,
                 pad_audio_to=None,
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
        self.pad_to = pad_audio_to
        super().__init__(standardize=standardize)

    def make_features(self, audio: np.ndarray) -> np.ndarray:
        """ Extract (log) spectrogram """

        # dither
        if self.dither > 0:
            audio = audio_utils.dither(audio, self.dither)

        # do preemphasis
        if self.preemph is not None:
            audio = audio_utils.preemphasize(audio, self.preemph)

        # pad if needed
        audio = self.pad(audio) if self.pad_to else audio

        # get spectrogram. Librosa returns (1 + n_fft/2, n_frames)
        # we need to transpose to have time as first dimension
        features = librosa.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length,
            center=True, window=self.window
        )
        # put features into correct order (n_frames, n_freqs)
        features = features.transpose()

        # get power spectrum
        if self.mag_power != 1.0:
            features = np.power(features, self.mag_power)

        # logarithm if needed
        if self.log:
            features = np.log(features + self.log_eps)

        # Leave only the spectrum we need
        features = features[:, :self.features_num]

        return features

    def pad(self, audio: np.ndarray) -> np.ndarray:
        """
        Padding signal is required if you need constant length features
        for TFlite"""
        length = 1 + int(
            (len(audio) - self.win_length) // self.hop_length + 1)
        pad_size = (self.pad_to - length % self.pad_to) * self.hop_length
        return np.pad(audio, (0, pad_size), mode='constant')
