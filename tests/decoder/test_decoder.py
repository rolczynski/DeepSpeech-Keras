import os
import numpy as np
import pytest
import automatic_speech_recognition as asr


@pytest.fixture
def decoder() -> asr.decoder.GreedyDecoder:
    return asr.decoder.GreedyDecoder()


def test_call(decoder):
    data = np.array([
        [[0.6, 0.2, 0.1, 0.1],
         [0.1, 0.1, 0.7, 0.1],
         [0.2, 0.4, 0.3, 0.1],
         [0.2, 0.4, 0.3, 0.1]],
        [[0.6, 0.2, 0.1, 0.1],
         [0.2, 0.4, 0.3, 0.1],
         [0.1, 0.1, 0.7, 0.1],
         [0.2, 0.4, 0.3, 0.1]]
    ]).astype(np.float32)
    labels_1, labels_2 = decoder(data)
    assert np.all(labels_1 == [0, 2, 1])
    assert np.all(labels_2 == [0, 1, 2, 1])


def test_save_load(decoder):
    file_path = 'decoder.bin'
    asr.utils.save(decoder, file_path)
    del decoder
    decoder = asr.utils.load(file_path)
    test_call(decoder)
    os.remove(file_path)
