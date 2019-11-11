import os
import numpy as np
import pytest
import automatic_speech_recognition as asr


@pytest.fixture
def decoder() -> asr.decoder.TensorflowDecoder:
    return asr.decoder.TensorflowDecoder(beam_size=100)


def test_call(decoder):
    data = np.array([
        [[0.6, 0.2, 0.1, 0.1],
         [0.1, 0.1, 0.7, 0.1],
         [0.2, 0.4, 0.3, 0.1],
         [0.2, 0.4, 0.3, 0.1]],

    ]).astype(np.float32)
    batch_labels = decoder(data).numpy()
    assert np.all(batch_labels == [[0, 2, 1]])      # The last column is blank, and last raw is not taken


def test_save_load(decoder):
    file_path = 'decoder.bin'
    asr.utils.save(decoder, file_path)
    del decoder
    decoder = asr.utils.load(file_path)
    assert decoder.beam_size == 100
    test_call(decoder)
    os.remove(file_path)
