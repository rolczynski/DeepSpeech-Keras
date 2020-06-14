import math
import numpy as np
import librosa
from . import audio_utils
from .. import features


class MFCC(features.FeaturesExtractor):
    """
    This class calculates the Mel-frequency Cepstral Coefficients (MFCCs)
    The procedure is described in:
    https://haythamfayek.com/2016/04/21/
    speech-processing-for-machine-learning.html
    """
    def __init__(self, features_num: int, sample_rate: int,
                 winlen: float = 0.02, winstep: float = 0.01,
                 window="hann", n_fft=None,
                 dct_type: int = 2, lifter: float = 0,
                 dither: float = 1e-5, preemph: float = 0.97,
                 standardize="per_feature"):

        self.features_num = features_num
        self.sample_rate = sample_rate
        self.win_length = math.ceil(winlen * sample_rate)
        self.hop_length = math.ceil(winstep * sample_rate)
        self.window = window
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.dct_type = dct_type
        self.lifter = lifter
        self.dither = dither
        self.preemph = preemph
        super().__init__(standardize=standardize)

    def make_features(self, audio: np.ndarray) -> np.ndarray:
        """ Extract MFCCs from the audio. """
        # dither
        if self.dither > 0:
            audio = audio_utils.dither(audio, self.dither)

        # do preemphasis
        if self.preemph is not None:
            audio = audio_utils.preemphasize(audio, self.preemph)

        # get mfccs. Librosa returns (n_mfcc, t)
        features = librosa.feature.mfcc(
            audio, self.sample_rate, n_mfcc=self.features_num,
            dct_type=self.dct_type, lifter=self.lifter, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.win_length,
            window=self.window
        )

        # put features into correct order (time, n_features)
        return features.transpose()
