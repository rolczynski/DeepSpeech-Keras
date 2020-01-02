import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision


def get_deepspeech2(input_dim, output_dim,
                    is_mixed_precision=True,
                    rnn_units=800, random_state=1) -> keras.Model:
    """

    input_dim: int i wielokrotność 4
    output_dim: licba liter w słowniku

    """
    if is_mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    # Create model under CPU scope and avoid OOM, errors during concatenation
    # a large distributed model.
    with tf.device('/cpu:0'):
        # Define input tensor [batch, time, features]
        input_tensor = layers.Input([None, input_dim], name='X')

        # Add 4th dimension [batch, time, frequency, channel]
        x = layers.Lambda(keras.backend.expand_dims,
                          arguments=dict(axis=-1))(input_tensor)
        x = layers.Conv2D(filters=32,
                          kernel_size=[11, 41],
                          strides=[2, 2],
                          padding='same',
                          use_bias=False,
                          name='conv_1')(x)
        x = layers.BatchNormalization(name='conv_1_bn')(x)
        x = layers.ReLU(name='conv_1_relu')(x)

        x = layers.Conv2D(filters=32,
                          kernel_size=[11, 21],
                          strides=[1, 2],
                          padding='same',
                          use_bias=False,
                          name='conv_2')(x)
        x = layers.BatchNormalization(name='conv_2_bn')(x)
        x = layers.ReLU(name='conv_2_relu')(x)
        # We need to squeeze to 3D tensor. Thanks to the stride in frequency
        # domain, we reduce the number of features four times for each channel.
        x = layers.Reshape([-1, input_dim//4*32])(x)

        for i in [1, 2, 3, 4, 5]:
            recurrent = layers.GRU(units=rnn_units,
                                   activation='tanh',
                                   recurrent_activation='sigmoid',
                                   use_bias=True,
                                   return_sequences=True,
                                   reset_after=True,
                                   name=f'gru_{i}')
            x = layers.Bidirectional(recurrent,
                                     name=f'bidirectional_{i}',
                                     merge_mode='concat')(x)
            x = layers.Dropout(rate=0.5)(x) if i < 5 else x  # Only between

        # Return at each time step logits along characters. Then CTC
        # computation is more stable, in contrast to the softmax.
        x = layers.TimeDistributed(layers.Dense(units=rnn_units*2), name='dense_1')(x)
        x = layers.ReLU(name='dense_1_relu')(x)
        x = layers.Dropout(rate=0.5)(x)
        output_tensor = layers.TimeDistributed(layers.Dense(units=output_dim),
                                               name='dense_2')(x)

        model = keras.Model(input_tensor, output_tensor, name='DeepSpeech2')
    return model
