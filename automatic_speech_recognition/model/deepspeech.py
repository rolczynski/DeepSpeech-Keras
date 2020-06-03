import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from automatic_speech_recognition.utils import load_graph_from_gfile
from collections import OrderedDict
import logging

logger = tf.get_logger()
logger.setLevel(logging.WARNING)


class ResetMask(layers.Layer):
    def __init__(self, **kwargs):
        super(ResetMask, self).__init__(**kwargs)
        self.supports_masking = True
        self._compute_output_and_mask_jointly = True

    def compute_mask(self, inputs, mask=None):
        return inputs[1]

    def call(self, inputs):
        # Compute the mask and outputs simultaneously.
        inputs[0]._keras_mask = inputs[1]
        return inputs[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def get_deepspeech(input_dim, output_dim, context=9, units=2048,
                   dropouts=(0.05, 0.05, 0.05, 0, 0.05),
                   tflite_version: bool = False,
                   random_state=1) -> keras.Model:
    """
    The `get_deepspeech` returns the graph definition of the DeepSpeech
    model. Default parameters are overwritten only where it is needed.

    Reference:
    "Deep Speech: Scaling up end-to-end speech recognition."
    (https://arxiv.org/abs/1412.5567)
    """
    if dropouts[3] != 0:
        logger.warning("Mozilla DeepSpeech doesn't use dropout "
                       "after LSTM(dropouts[3]). Be careful!")
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    max_seq_length = None
    if tflite_version:
        max_seq_length = 3

    with tf.device('/cpu:0'):
        input_tensor = layers.Input([max_seq_length, input_dim], name='X')

        # Add 4th dimension [batch, time, frequency, channel]
        x = layers.Lambda(keras.backend.expand_dims,
                          arguments=dict(axis=3))(input_tensor)
        # Fill zeros around time dimension
        x = layers.ZeroPadding2D(padding=(context, 0))(x)
        # Convolve signal in time dim
        receptive_field = (2 * context + 1, input_dim)
        x = layers.Conv2D(filters=units, kernel_size=receptive_field)(x)
        # Squeeze into 3rd dim array
        x = layers.Lambda(keras.backend.squeeze, arguments=dict(axis=2))(x)

        x = layers.ReLU()(x)
        x = layers.Dropout(rate=dropouts[0])(x)

        x = layers.TimeDistributed(
            layers.Dense(units), name='dense_2')(x)
        x = layers.ReLU(max_value=20)(x)
        x = layers.Dropout(rate=dropouts[1])(x)

        x = layers.TimeDistributed(
            layers.Dense(units), name='dense_3')(x)
        x = layers.ReLU(max_value=20)(x)
        x = layers.Dropout(rate=dropouts[2])(x)

        x = layers.LSTM(units, return_sequences=True,
                        name='lstm_1', unroll=tflite_version)(x)
        x = layers.Dropout(rate=dropouts[3])(x)

        x = layers.TimeDistributed(
            layers.Dense(units), name='dense_4')(x)
        x = layers.ReLU(max_value=20)(x)
        x = layers.Dropout(rate=dropouts[4])(x)

        x = layers.TimeDistributed(
            layers.Dense(output_dim), name='dense_5')(x)

        if not tflite_version:
            # mask parts of the input to preserve information
            # about actual (non-padded) sequence length
            x = ResetMask()([
                x, tf.reduce_any(input_tensor != 0.0, axis=-1)])

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


def load_mozilla_deepspeech(path="./data/output_graph.pb", tflite_version=False):
    loaded_tensors, loaded_graph = load_graph_from_gfile(path)
    loaded_weights = []
    for key in loaded_tensors.keys():
        # check if tensor really represents a weight tensor
        if loaded_tensors[key].size > 10 and 'Const' not in key:
            print(
                f'Found weight tensor {key} with '
                f'shape {loaded_tensors[key].shape}')
            loaded_weights.append(loaded_tensors[key])

    # Fix differences in stored weights between mozilla deepspeech and keras
    W_x, W_h, b = reformat_deepspeech_lstm(loaded_weights[6], loaded_weights[7])
    # TODO try using convolution to avoid materializing bigger matrix
    loaded_weights[1] = loaded_weights[1].reshape((19, 26, 1, 2048))

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
