import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ..utils import load_deepspeech_graph
from collections import OrderedDict
import logging

logger = tf.get_logger()
logger.setLevel(logging.WARNING)


def create_overlapping_windows(batch_x, num_channels, context=9, return_stacked=True):
    batch_size = tf.shape(input=batch_x)[0]
    window_width = 2 * context + 1

    # Create a constant convolution filter using an identity matrix, so that the
    # convolution returns patches of the input tensor as is, and we can create
    # overlapping windows over the MFCCs.
    eye_filter = tf.constant(np.eye(window_width * num_channels)
                             .reshape(window_width, num_channels, window_width * num_channels),
                             tf.float32)  # pylint: disable=bad-continuation

    # Create overlapping windows
    batch_x = tf.nn.conv1d(input=batch_x, filters=eye_filter, stride=1, padding='SAME')

    # Remove dummy depth dimension and reshape into [batch_size, n_windows, window_width, n_input]
    if return_stacked:
        batch_x = tf.reshape(batch_x, [batch_size, -1, window_width * num_channels])
    else:
        batch_x = tf.reshape(batch_x, [batch_size, -1, window_width, num_channels])

    return batch_x


def get_deepspeech(input_dim, output_dim, context=9, units=2048,
                   dropouts=(0.05, 0.05, 0.05, 0, 0.05), tflite_version: bool = False,
                   random_state=1) -> keras.Model:
    """
    The `get_deepspeech` returns the graph definition of the DeepSpeech
    model. Then simple architectures like this can be easily serialize.
    Default parameters are overwrite only wherein it is needed.

    Reference:
    "Deep Speech: Scaling up end-to-end speech recognition."
    (https://arxiv.org/abs/1412.5567)
    """
    if dropouts[3] != 0:
        logger.warning("Mozilla DeepSpeech doesn't use dropout after LSTM(dropouts[3]). Be careful!")
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    max_seq_length = None
    if tflite_version:
        max_seq_length = 3

    with tf.device('/cpu:0'):
        input_tensor = layers.Input([max_seq_length, input_dim], name='X')
        x = create_overlapping_windows(input_tensor, input_dim, context)
        # create overlapping windows loses shape data. Reshape restores it.
        x = layers.Reshape([max_seq_length if max_seq_length else -1, (2 * context + 1) * input_dim])(x)
        x = layers.Dense(units)(x)

        x = layers.ReLU()(x)
        x = layers.Dropout(rate=dropouts[0])(x)

        x = layers.Dense(units)(x)
        x = layers.ReLU(max_value=20)(x)
        x = layers.Dropout(rate=dropouts[1])(x)

        x = layers.Dense(units)(x)
        x = layers.ReLU(max_value=20)(x)
        x = layers.Dropout(rate=dropouts[2])(x)

        x = layers.LSTM(units, return_sequences=True, unroll=tflite_version)(x)
        x = layers.Dropout(rate=dropouts[3])(x)

        x = layers.Dense(units)(x)
        x = layers.ReLU(max_value=20)(x)
        x = layers.Dropout(rate=dropouts[4])(x)

        x = layers.Dense(output_dim)(x)

        if tflite_version:
            model = keras.Model(input_tensor, x, name='DeepSpeech')
        else:
            # Having 1 element vector is required to save and load model in non nightly tensorflow
            # https://github.com/tensorflow/tensorflow/issues/35446.
            model = keras.Model(input_tensor, x, name='DeepSpeech')
    return model


def reformat_deepspeech_lstm(W, b):
    """
    Deepspeech lstm weights are 2 tensors: stacked weights and biases respectively. This function cuts those
    tensors to fit keras weight format.
    :param W: Weights of deepspeech lstm tensor
    :param b: biases of deepspeech lstm tensor
    :return: (W_x, W_h, b)
    """
    w_i, w_f, w_C, w_o = np.split(W, 4, axis=1)
    w_xi = w_i[:2048]
    w_hi = w_i[2048:]
    w_xf = w_f[:2048]
    w_hf = w_f[2048:]
    w_xC = w_C[:2048]
    w_hC = w_C[2048:]
    w_xo = w_o[:2048]
    w_ho = w_o[2048:]

    b_i, b_f, b_C, b_o = np.split(b, 4, axis=0)
    W_x = np.hstack((w_xi, w_xC, w_xf, w_xo))
    W_h = np.hstack((w_hi, w_hC, w_hf, w_ho))
    b = np.hstack((b_i, b_C, b_f, b_o))
    return W_x, W_h, b


def load_mozila_deepspeech(path="./data/output_graph.pb", tflite_version=False):
    loaded_tensors, loaded_graph = load_deepspeech_graph(path)
    loaded_weights = []
    for key in loaded_tensors.keys():
        # check if tensor really represents a weight tensor
        if loaded_tensors[key].size > 10 and 'Const' not in key:
            print(f'Found weight tensor {key} with shape {loaded_tensors[key].shape}')
            loaded_weights.append(loaded_tensors[key])

    # Fix differences in stored weights between mozilla deepspeech and keras
    W_x, W_h, b = reformat_deepspeech_lstm(loaded_weights[6], loaded_weights[7])
    # TODO try using convcolution to avoid materializing bigger matrix
    # w[1] = w[1].reshape((26, 19, 1, 2048))

    keras_weights = [
        loaded_weights[1], loaded_weights[0],  # Dense 1
        loaded_weights[3], loaded_weights[2],  # Dense 2
        loaded_weights[5], loaded_weights[4],  # Dense 3
        W_x, W_h, b,  # LSTM
        loaded_weights[9], loaded_weights[8],  # Dense 4
        loaded_weights[11], loaded_weights[10]  # Dense 5
    ]
    print("Shapes of weights prepared to be loaded into keras model")
    print([w.shape for w in keras_weights])

    # Deepspeech specs are taken from Mozilla Deepspeech
    model = get_deepspeech(input_dim=26,
                           output_dim=29,
                           context=9,
                           units=2048,
                           dropouts=(0, 0, 0, 0, 0),
                           tflite_version=tflite_version)
    model.set_weights(keras_weights)
    return model
