import pytest
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from testfixtures import LogCapture
from unittest import mock
import automatic_speech_recognition as asr


@pytest.fixture
def pipeline() -> asr.pipeline.CTCPipeline:
    # Use graph definition, because target tensors can no be defined
    input_tensor = layers.Input([None, 3], name='X')
    output_tensor = layers.TimeDistributed(layers.Dense(2))(input_tensor)
    model = keras.Model(input_tensor, output_tensor, name='Test-Model')
    optimizer = keras.optimizers.SGD(learning_rate=.1)
    return asr.pipeline.CTCPipeline(
        alphabet=mock.Mock(),
        model=model,
        optimizer=optimizer,
        decoder=mock.Mock(),
        features_extractor=mock.Mock()
    )


def test_preprocess_data(pipeline):
    batch = ([np.zeros([10])], ['transcript'])
    pipeline.features_extractor.return_value = 'features'
    pipeline.alphabet.get_batch_labels = mock.Mock()
    pipeline.alphabet.get_batch_labels.return_value = 'labels'
    assert pipeline.preprocess(batch) == ('features', 'labels')


def test_compile_model(pipeline):
    features = np.random.random([1, 10, 3]).astype(np.float32)
    labels = np.array([[0, 1]], dtype=np.int16)

    with pytest.raises(RuntimeError):
        pipeline.model.fit(features, labels)

    with LogCapture() as log:
        pipeline.compile_model()
    record = next(record for record in log.records
                  if record.name == 'asr.pipeline')
    assert record.msg == "Model is successfully compiled"

    # Now, you should be able to fit the model
    # To preview values directly in the loss function:
    #   pipeline.model.run_eagerly = True
    history_cb = pipeline.model.fit(features, labels)
    assert isinstance(history_cb, callbacks.History)


def test_fit(pipeline):
    features = np.random.random([1, 10, 3]).astype(np.float32)
    labels = np.array([[0, 1]], dtype=np.int16)
    dataset = ((features, labels) for i in range(100))  # build simple generator

    # Skip pre-processing stage
    pipeline.wrap_preprocess = mock.Mock()
    pipeline.wrap_preprocess.return_value = dataset
    # Fake dev dataset
    dev_dataset = dataset

    history_cb = pipeline.fit(dataset, dev_dataset, epochs=2,
                              steps_per_epoch=10, validation_steps=1)
    assert isinstance(history_cb, callbacks.History)
    loss_epoch_1, loss_epoch_2 = history_cb.history['loss']
    assert loss_epoch_1 > loss_epoch_2


def test_distribute_model():
    model = mock.Mock()
    gpus = mock.MagicMock()     # default __len__ method

    with mock.patch('tensorflow.keras.utils') as mock_keras, LogCapture() as log:
        mock_keras.multi_gpu_model = mock.Mock()
        mock_keras.multi_gpu_model.side_effect = ValueError
        dist_model = asr.pipeline.CTCPipeline.distribute_model(model, gpus)
        assert dist_model == model

    record = next(record for record in log.records
                  if record.name == 'asr.pipeline')
    assert record.msg == "Training using single GPU or CPU"


def test_get_loss():
    loss = asr.pipeline.CTCPipeline.get_loss()
    y = [[0, 1]]
    logits_random = np.random.random([1, 5, 3]).astype(np.float32)
    ctc_loss_random = loss(y, logits_random)
    assert ctc_loss_random.shape == (1,)

    # This is 0011. The last column represents the blank token.
    logits = np.array([[[1, -1, -1],
                        [1, -1, -1],
                        [-1, 1, -1],
                        [-1, 1, -1]]], dtype=np.float32) * 100
    ctc_loss = loss(y, logits)
    ctc_loss_value, = np.around(ctc_loss.numpy(), decimals=3)
    assert ctc_loss_value == 0
