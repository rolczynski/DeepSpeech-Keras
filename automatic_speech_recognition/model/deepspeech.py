import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_deepspeech(input_dim, output_dim, context=7, units=1024,
                   dropouts=(0.1, 0.1, 0), random_state=1) -> keras.Model:
    """
    The `get_deepspeech` returns the graph definition of the DeepSpeech
    model. Then simple architectures like this can be easily serialize.
    Default parameters are overwrite only wherein it is needed.

    Reference:
    "Deep Speech: Scaling up end-to-end speech recognition."
    (https://arxiv.org/abs/1412.5567)
    """
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
        # Fill zeros around time dimension
        x = layers.ZeroPadding2D(padding=(context, 0))(x)
        # Convolve signal in time dim
        receptive_field = (2 * context + 1, input_dim)
        x = layers.Conv2D(filters=units, kernel_size=receptive_field)(x)
        # Squeeze into 3rd dim array
        x = layers.Lambda(keras.backend.squeeze, arguments=dict(axis=2))(x)
        # Add non-linearity
        x = layers.ReLU(max_value=20)(x)
        # Use dropout as regularization
        x = layers.Dropout(rate=dropouts[0])(x)

        # 2nd and 3rd FC layers do a feature extraction base on a narrow
        # context of convolutional layer
        x = layers.TimeDistributed(layers.Dense(units))(x)
        x = layers.ReLU(max_value=20)(x)
        x = layers.Dropout(rate=dropouts[1])(x)

        x = layers.TimeDistributed(layers.Dense(units))(x)
        x = layers.ReLU(max_value=20)(x)
        x = layers.Dropout(rate=dropouts[2])(x)

        # Use recurrent layer to have a broader context
        x = layers.Bidirectional(layers.LSTM(units, return_sequences=True),
                                 merge_mode='sum')(x)

        # Return at each time step logits along characters. Then CTC
        # computation is more stable, in contrast to the softmax.
        output_tensor = layers.TimeDistributed(layers.Dense(output_dim))(x)
        model = keras.Model(input_tensor, output_tensor, name='DeepSpeech')
    return model
