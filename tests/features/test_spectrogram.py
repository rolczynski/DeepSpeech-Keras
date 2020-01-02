import os
import numpy as np
import pytest
import automatic_speech_recognition as asr


@pytest.fixture
def feature_extractor() -> asr.features.Spectrogram:
    return asr.features.Spectrogram(
        features_num=80,
        samplerate=16000,
        winlen=0.025,
        winstep=0.01,
        winfunc=np.hamming
    )


def test_make_features(feature_extractor):
    directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(directory, '../sample-en.wav')
    audio = asr.utils.read_audio(file_path)
    features = feature_extractor.make_features(audio)
    assert features.dtype == np.float64
    assert features.shape == (404, 80)


def test_save_load(feature_extractor):
    file_path = 'feature_extractor.bin'
    asr.utils.save(feature_extractor, file_path)
    del feature_extractor
    feature_extractor = asr.utils.load(file_path)
    test_make_features(feature_extractor)
    os.remove(file_path)
