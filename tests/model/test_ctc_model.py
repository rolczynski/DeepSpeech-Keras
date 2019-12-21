import numpy as np
import pytest
import automatic_speech_recognition as asr


@pytest.fixture
def model() -> asr.model.get_ctc_model:
    return asr.model.get_ctc_model(
        layers_params=[{
            'constructor': 'expand_dims',
            'axis': -1
        }, {
            'constructor': 'ZeroPadding2D',
            'padding': [7, 20]
        }, {
            'constructor': 'Conv2D',
            'name': 'base_1',
            'filters': 2,
            'kernel_size': [15, 41],
            'strides': [2, 2]
        }, {
            'constructor': 'squeeze_last_dims',
            'units': 80
        }, {
            'constructor': 'LSTM',
            'name': 'base_2',
            'units': 10,
            'return_sequences': True
        }, {
            'constructor': 'Dense',
            'name': 'base_3',
            'units': 36,
            'activation': 'softmax'
        }],
        input_dim=80
    )


def test_ctc_model(model):
    # print(model.summary())
    X = np.random.random([7, 10, 80]).astype(np.float32)
    y_hat = model(X)
    assert y_hat.shape == (7, 5, 36)
