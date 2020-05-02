import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision


def reduce_time(outputs, factor=2, time_major=True):
    """
    Taken from https://github.com/thomasschmied/Speech_Recognition_with_Tensorflow/blob/master/SpeechRecognizer.py
    Reshapes the given outputs, i.e. reduces the
    time resolution by 2.
    Similar to "Listen Attend Spell".
    https://arxiv.org/pdf/1508.01211.pdf
    """
    assert time_major
    assert factor == 2
    # [max_time, batch_size, num_units]
    shape = tf.shape(outputs)
    max_time, batch_size, num_units = outputs.shape
    # if static dimension is None use runtime Tensor value
    if max_time is None:
        max_time = shape[0]
    if batch_size is None:
        batch_size = shape[1]
    if num_units is None:
        raise ValueError("Last dimension of input tensor should be known")

    pads = [[0, tf.math.floormod(max_time, 2)], [0, 0], [0, 0]]
    outputs = tf.pad(outputs, pads)
    concat_outputs = tf.reshape(outputs, (-1, batch_size, num_units * 2))
    return concat_outputs


class TimeReductionLayer(keras.layers.Layer):
    def __init__(self, factor, time_major):
        super().__init__()
        assert factor == 2
        assert time_major
        self._factor = factor
        self._time_major = time_major

    def call(self, inputs, **kwargs):
        return reduce_time(inputs, self._factor, self._time_major)

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError("Input tensor should have 3 dimensions [max_time, batch_size, num_units].")
        max_time, batch_size, num_units = input_shape
        if num_units is None:
            raise ValueError("Last dimension of input tensor should be known")
        output_max_time = None if max_time is None else (max_time // 2 + max_time % 2)
        _output_shape = (output_max_time, batch_size, num_units)
        return _output_shape

def encoder(num_features,
            batch_size,
            num_layers,
            d_model,
            reduction_index,
            convert_tflite=False):
    max_seq_length = None
    if convert_tflite:
        max_seq_length = 50

    encoder_input = layers.Input([max_seq_length, num_features], batch_size=batch_size)
    x = tf.transpose(encoder_input, [1, 0, 2])
    for i in range(num_layers):
        x = layers.LSTM(d_model // 2,
                        return_sequences=True,
                        time_major=True,
                        stateful=True,
                        unroll=convert_tflite)(x)

        # TODO do something with layer normalisation
        x = tf.keras.layers.LayerNormalization()(x)

        if i == reduction_index:
            # Warning originally there was a wrong time reduction incompatible with time_major tensors
            x = TimeReductionLayer(factor=2, time_major=True)(x)

    x = tf.transpose(x, [1, 0, 2])
    return keras.Model(encoder_input, x, name='encoder')


def prediction_network(batch_size,
                       embedding_size,
                       num_layers,
                       layer_size,
                       convert_tflite=False):
    max_seq_length = None
    if convert_tflite:
        max_seq_length = 50
    embedded_labels = tf.keras.Input(shape=[max_seq_length, embedding_size], batch_size=batch_size)

    x = tf.transpose(embedded_labels, [1, 0, 2])
    for _ in range(num_layers):
        x = layers.LSTM(layer_size,
                        return_sequences=True,
                        time_major=True,
                        stateful=True,
                        unroll=convert_tflite)(x)
        x = layers.LayerNormalization()(x)
    x = tf.transpose(x, [1, 0, 2])
    return keras.Model(embedded_labels, x, name='pred_net')


def get_rnnt(input_dim,
             batch_size,
             num_layers_encoder=2,
             rnn_units_encoder=400,
             reduction_index_encoder=0,
             num_layers_pred=2,
             embed_size_pred=50,
             vocab_size_pred=10,
             convert_tflite=False, random_state=10) -> keras.Model:
    max_seq_length = None
    if convert_tflite:
        max_seq_length = 50

    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    # Create model under CPU scope and avoid OOM, errors during concatenation
    # a large distributed model.
    # Define input tensor [batch, time, features]
    input_tensor = layers.Input([max_seq_length, input_dim], batch_size=batch_size, name='features')
    labels = tf.keras.Input(shape=[max_seq_length], batch_size=batch_size, dtype=tf.int32, name='labels')

    inp_enc = encoder(
        input_dim,
        batch_size=batch_size,
        num_layers=num_layers_encoder,
        d_model=rnn_units_encoder,
        reduction_index=reduction_index_encoder,
        convert_tflite=convert_tflite)(input_tensor)

    embedded_labels = layers.Embedding(vocab_size_pred, embed_size_pred, name='embedding')(labels)
    # Loss requires first step to be zero vector
    embedded_labels = tf.pad(embedded_labels, [[0, 0], [1, 0], [0, 0]], constant_values=0)
    pred_outputs = prediction_network(
        batch_size=batch_size,
        num_layers=num_layers_pred,
        embedding_size=embed_size_pred,
        layer_size=rnn_units_encoder // 2,
        convert_tflite=convert_tflite)(embedded_labels)

    joint_inp = tf.concat([
        tf.tile(tf.expand_dims(inp_enc, 2), [1, 1, tf.shape(pred_outputs)[1], 1]),  # [B, T, V] => [B, T, 1, V]
        tf.tile(tf.expand_dims(pred_outputs, 1), [1, tf.shape(inp_enc)[1], 1, 1])  # [B, U, V] => [B, 1, U, V]
    ], axis=-1)

    joint_outputs = tf.keras.layers.Dense(vocab_size_pred, name='joint')(joint_inp)
    outputs = tf.nn.log_softmax(joint_outputs, axis=-1)

    if convert_tflite:
        return keras.Model([input_tensor, labels], outputs, name='RNNT')
    else:
        # Having 1 element vector is required to save and load model in non nightly tensorflow
        # https://github.com/tensorflow/tensorflow/issues/35446.
        feature_lengths = tf.keras.Input(shape=[1], dtype=tf.int32, name='feature_lengths')
        label_lengths = tf.keras.Input(shape=[1], dtype=tf.int32, name='label_lengths')
        return keras.Model([input_tensor, labels, feature_lengths, label_lengths], outputs, name='RNNT')
