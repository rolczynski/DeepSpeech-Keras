import os
import numpy as np
import pytest
import automatic_speech_recognition as asr


@pytest.fixture
def feature_extractor() -> asr.features.FilterBanks:
    return asr.features.FilterBanks(
        winlen=0.025,
        winstep=0.01,
        nfilt=80,
        winfunc='hamming'
    )


def test_make_features(feature_extractor):
    directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(directory, f'sample.wav')
    audio = asr.utils.read_audio(file_path)
    features = feature_extractor.make_features(audio)
    assert features.dtype == np.float64
    assert features.shape == (450, 80)
