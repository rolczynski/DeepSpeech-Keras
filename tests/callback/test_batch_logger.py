import os
import numpy as np
from testfixtures import LogCapture
import automatic_speech_recognition as asr


def test_batch_logger():
    file_path = 'test.bin'
    with LogCapture() as log:
        callback = asr.callback.BatchLogger(file_path)
        callback.on_epoch_begin()
        values = np.random.random([100])
        for i, value in enumerate(values):
            callback.on_train_batch_end(i, {'loss': value})
        callback.on_epoch_end(epoch=0, logs=dict(loss=0, val_loss=1))

    results = asr.utils.load(file_path)
    assert results[0][:-1] == [0, 0, 1]
    assert np.array_equal(results[0][-1], values)
    os.remove(file_path)

    assert len(log.records) == 101
    assert log.records[0].name == 'asr.callback'
