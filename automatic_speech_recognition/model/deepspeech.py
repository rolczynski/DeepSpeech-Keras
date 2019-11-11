import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def DeepSpeech(input_dim=80, output_dim=36, context=7, units=1024,
               dropouts=(0.1, 0.1, 0), random_state=1) -> keras.Model:
    """
    Model is adapted from: Deep Speech: Scaling up end-to-end speech recognition.
    Default parameters are overwrite wherein it is needed.

    Reference:
    https://arxiv.org/abs/1412.5567
    """
    np.random.seed(random_state)
    tf.random.set_seed(random_state)                                                            # Create model under CPU scope and avoid OOM
    with tf.device('/cpu:0'):                                                                   # erors during concatenation a large distributed model.
        input_tensor = layers.Input([None, input_dim], name='X')                                # Define input tensor [time, features]
        x = layers.Lambda(keras.backend.expand_dims, arguments=dict(axis=-1))(input_tensor)     # Add 4th dim (channel)
        x = layers.ZeroPadding2D(padding=(context, 0))(x)                                       # Fill zeros around time dimension
        receptive_field = (2*context + 1, input_dim)                                            # Take into account fore/back-ward context
        x = layers.Conv2D(filters=units, kernel_size=receptive_field)(x)                        # Convolve signal in time dim
        x = layers.Lambda(keras.backend.squeeze, arguments=dict(axis=2))(x)                     # Squeeze into 3rd dim array
        x = layers.ReLU(max_value=20)(x)                                                        # Add non-linearity
        x = layers.Dropout(rate=dropouts[0])(x)                                                 # Use dropout as regularization

        x = layers.TimeDistributed(layers.Dense(units))(x)                                      # 2nd and 3rd FC layers do a feature
        x = layers.ReLU(max_value=20)(x)                                                        # extraction base on the context
        x = layers.Dropout(rate=dropouts[1])(x)

        x = layers.TimeDistributed(layers.Dense(units))(x)
        x = layers.ReLU(max_value=20)(x)
        x = layers.Dropout(rate=dropouts[2])(x)

        x = layers.Bidirectional(layers.LSTM(units, return_sequences=True), merge_mode='sum')(x)

        output_tensor = layers.TimeDistributed(layers.Dense(output_dim, activation='softmax'))(x)  # Return at each time step prob along characters
        model = keras.Model(input_tensor, output_tensor, name='DeepSpeech')
    return model
