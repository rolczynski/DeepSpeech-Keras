import os
from unittest.mock import MagicMock
import numpy as np
from testfixtures import LogCapture
import automatic_speech_recognition as asr
np.random.seed(1)


def test_distributed_checkpoint():
    model = MagicMock()
    model.save = MagicMock()
    model.load_weights = MagicMock()
    log_dir = 'test_distributed_checkpoint'

    with LogCapture() as log:
        callback = asr.callback.DistributedModelCheckpoint(model, log_dir)
        callback.on_train_begin()
        values = np.random.random([20, 2])
        for i, (x1, x2) in enumerate(values):
            callback.on_epoch_end(i, {'loss': x1, 'val_loss': x2})
        callback.on_train_end()

    os.removedirs(log_dir)
    assert np.isclose(callback.best_result, 0.01, atol=0.01)

    assert len(log.records) == 4
    assert log.records[0].name == 'asr.callback'
