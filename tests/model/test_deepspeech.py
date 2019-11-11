import numpy as np
import pytest
import automatic_speech_recognition as asr


@pytest.fixture
def model() -> asr.model.DeepSpeech:
    return asr.model.DeepSpeech(
        input_dim=80,
        output_dim=36,
        context=7,
        units=128
    )


def test_deepspeech(model):
    # print(model.summary())
    X = np.random.random([7, 10, 80]).astype(np.float32)
    y_hat = model(X)
    assert y_hat.shape == (7, 10, 36)
