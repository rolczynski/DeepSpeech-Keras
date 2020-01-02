import h5py
import numpy as np
import automatic_speech_recognition as asr


def transfer_open_seq2seq_deepspeech2(weights_file_name: str, store_file_name: str):
    """ To transfer model you should extracted weights (and activations for
    testing purpose) in the original Open Seq2Seq implementation. """
    source_weights = asr.utils.load(weights_file_name)
    w = change_names(source_weights)
    layer_names = ['conv_1', 'conv_1_bn', 'conv_2', 'conv_2_bn',
                   'bidirectional_1', 'bidirectional_2', 'bidirectional_3',
                   'bidirectional_4', 'bidirectional_5', 'dense_1', 'dense_2']

    with h5py.File(store_file_name, 'w') as store:

        # Convolution Blocks
        for i in [1, 2]:

            # Convolutional Layer
            layer = store.create_group(f'conv_{i}')
            key = f'conv_{i}/kernel'
            name = f'{key}:0'
            layer[name] = w[key]
            layer.attrs['weight_names'] = [str.encode(name)]

            # Bach Normalization
            layer = store.create_group(f'conv_{i}_bn')
            layer_weights = []
            for param in ['gamma', 'beta', 'moving_mean', 'moving_variance']:
                key = f'conv_{i}_bn/{param}'
                name = f'{key}:0'
                layer[name] = w[key]
                layer_weights.append(name)
            layer.attrs['weight_names'] = [str.encode(name) for name in layer_weights]


        # Recurrent Layers
        x_dim = 1280
        units = 800
        for i in [1, 2, 3, 4, 5]:
            layer = store.create_group(f'bidirectional_{i}')
            layer_weights = []

            for d in ['forward', 'backward']:

                # For the first layer, kernel shape == [2080, 1600]
                kernel = w[f'bidirectional_{i}/{d}']['kernel']
                # We have to divide kernel into four pieces
                kernel_r = kernel[:-units, :units]  # Upper left
                kernel_z = kernel[:-units, units:]  # Upper right
                recurrent_kernel_r = kernel[-units:, :units]  # Bottom left
                recurrent_kernel_z = kernel[-units:, units:]  # Bottom right
                # and bias shape contains only two parts
                bias = w[f'bidirectional_{i}/{d}']['bias']
                bias_r, bias_z = bias[:units], bias[units:]

                # Rest parts are defined explicitly
                # For the first layer, kernel_h shape == [1280, 800]
                kernel_h = w[f'bidirectional_{i}/{d}']['kernel_h']
                bias_h = w[f'bidirectional_{i}/{d}']['bias_h']
                # For the first layer, kernel_h shape == [800, 800]
                recurrent_kernel_h = w[f'bidirectional_{i}/{d}']['recurrent_kernel_h']
                recurrent_bias_h = w[f'bidirectional_{i}/{d}']['recurrent_bias_h']

                # Now, we compose our layer weights format
                # For the first layer, kernel shape == [1280, 2400]
                kernel = np.zeros([x_dim, units * 3])
                kernel[:, :units] = kernel_z  # the r and z kernels are switched
                kernel[:, units:units * 2] = kernel_r
                kernel[:, units * 2:] = kernel_h
                # Now we define the names, which are understandable to Keras
                name = f'bidirectional_{i}/{d}_gru_{i}/kernel:0'
                layer[name] = kernel
                layer_weights.append(name)

                # For the first layer, recurrent_kernel shape == [800, 2400]
                kernel = np.zeros([units, units * 3])
                kernel[:, :units] = recurrent_kernel_z
                kernel[:, units:units * 2] = recurrent_kernel_r
                kernel[:, units * 2:] = recurrent_kernel_h
                name = f'bidirectional_{i}/{d}_gru_{i}/recurrent_kernel:0'
                layer[name] = kernel
                layer_weights.append(name)

                # The r and z biases are connected, bias shape == [2, 2400]
                bias = np.zeros([2, units * 3])
                bias[0, :units] = bias_z
                bias[0, units:units * 2] = bias_r
                bias[0, units * 2:] = bias_h
                bias[1, units * 2:] = recurrent_bias_h
                name = f'bidirectional_{i}/{d}_gru_{i}/bias:0'
                layer[name] = bias
                layer_weights.append(name)

            x_dim = units * 2  # Then the next layer consumes hidden states
            # Add references to layer weights
            layer.attrs['weight_names'] = [str.encode(n) for n in layer_weights]


        # Encoder Layers
        # kernel shape == [1600, 1600]
        layer = store.create_group('dense_1')
        layer['dense_1/kernel:0'] = w['dense_1/kernel']
        layer['dense_1/bias:0'] = w['dense_1/bias']
        layer.attrs['weight_names'] = [b'dense_1/kernel:0', b'dense_1/bias:0']

        # kernel shape == [1600, alphabet_size]
        layer = store.create_group('dense_2')
        layer['dense_2/kernel:0'] = w['dense_2/kernel']
        layer['dense_2/bias:0'] = w['dense_2/bias']
        layer.attrs['weight_names'] = [b'dense_2/kernel:0', b'dense_2/bias:0']


        # Register saved layers
        store.attrs['layer_names'] = [str.encode(name) for name in layer_names]
    return w


def change_names(weights):
    w = {}
    # Convolutional Layers
    for i in [1, 2]:
        key = f'ForwardPass/ds2_encoder/conv{i}/kernel:0'
        w[f'conv_{i}/kernel'] = weights[key]
        for param in ['gamma', 'beta', 'moving_mean', 'moving_variance']:
            key = f'ForwardPass/ds2_encoder/conv{i}/bn/{param}:0'
            w[f'conv_{i}_bn/{param}'] = weights[key]

    # Recurrent Layers
    for i in [1, 2, 3, 4, 5]:
        for original_d, d in [('fw', 'forward'), ('bw', 'backward')]:
            w[f'bidirectional_{i}/{d}'] = {}
            prefix = 'ForwardPass/ds2_encoder/cudnn_gru/' \
                     f'stack_bidirectional_rnn/cell_{i-1}' \
                     f'/bidirectional_rnn/{original_d}/cudnn_compatible_gru_cell'
            # Kernel
            key = prefix + '/gates/kernel'
            w[f'bidirectional_{i}/{d}']['kernel'] = weights[key]
            key = prefix + '/gates/bias'
            w[f'bidirectional_{i}/{d}']['bias'] = weights[key]

            # Input projection
            key = prefix + '/candidate/input_projection/kernel'
            w[f'bidirectional_{i}/{d}']['kernel_h'] = weights[key]
            key = prefix + '/candidate/input_projection/bias'
            w[f'bidirectional_{i}/{d}']['bias_h'] = weights[key]

            # Hidden projection
            key = prefix + '/candidate/hidden_projection/kernel'
            w[f'bidirectional_{i}/{d}']['recurrent_kernel_h'] = weights[key]
            key = prefix + '/candidate/hidden_projection/bias'
            w[f'bidirectional_{i}/{d}']['recurrent_bias_h'] = weights[key]

    # Encoder Layers
    prefix = 'ForwardPass/ds2_encoder/fully_connected'
    key = prefix + '/kernel:0'
    w['dense_1/kernel'] = weights[key]
    key = prefix + '/bias:0'
    w['dense_1/bias'] = weights[key]

    prefix = 'ForwardPass/fully_connected_ctc_decoder/fully_connected'
    key = prefix + '/kernel:0'
    w['dense_2/kernel'] = weights[key]
    key = prefix + '/bias:0'
    w['dense_2/bias'] = weights[key]
    return w
