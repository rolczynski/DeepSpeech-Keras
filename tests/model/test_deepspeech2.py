import numpy as np
import pytest
from tensorflow import keras
import automatic_speech_recognition as asr


@pytest.fixture
def model() -> keras.Model:
    return asr.model.get_deepspeech2(
        input_dim=160,
        output_dim=29
    )


def test_deepspeech2(model):
    X = np.random.random([7, 20, 160]).astype(np.float32)
    y_hat = model(X)
    # The output tensor [batch, time, features], where time is by half due to
    # the stride in time domain.
    assert y_hat.shape == (7, 10, 29)
