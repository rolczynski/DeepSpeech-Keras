import os
import h5py
import numpy as np
import automatic_speech_recognition as asr


def test_download_from_bucket():
    """
    # Before create a store, upload it manually
    with h5py.File('test-weights.h5', mode='w') as store:
        store['data'] = np.zeros([100, 10])
    """
    file_name = 'test-weights.h5'
    asr.utils.download_from_bucket(
        bucket_name='automatic-speech-recognition',
        remote_path=file_name,
        local_path=file_name
    )
    with h5py.File(file_name, mode='r') as store:
        data = store['data'][:]
    assert np.allclose(data, np.zeros([100, 10]))
    os.remove(file_name)
