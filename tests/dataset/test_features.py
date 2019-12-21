import os
import h5py
import pytest
import numpy as np
import pandas as pd
import automatic_speech_recognition as asr


@pytest.fixture
def dataset() -> asr.dataset.Features:
    file_path = 'test.h5'
    reference = pd.DataFrame({
        'path': [f'dataset/{i}' for i in range(10)],
        'transcript': [f'transcript-{i}' for i in range(10)],
    })

    with h5py.File(file_path, 'w') as store:
        for path in reference.path:
            store[path] = np.random.random([20, 10])

    with pd.HDFStore(file_path, mode='r+') as store:
        store['references'] = reference

    return asr.dataset.Features.from_hdf(file_path, batch_size=3)


def test_get_batch(dataset):
    batch_audio, transcripts = dataset.get_batch(index=1)
    a, b, c = transcripts
    assert b == 'transcript-4'
    a, b, c = batch_audio
    assert b.shape == (20, 10)
    # Remove store at the end of tests
    os.remove('test.h5')
