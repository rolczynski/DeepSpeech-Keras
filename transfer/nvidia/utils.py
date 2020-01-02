import numpy as np


def numpy_gru(x, w, units):
    """ This is the cuDNN compatible GRU (from CuDNN library user guide). """
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    time, features = x.shape
    h_last = np.zeros([units])

    for t in range(time):
        x_t = x[t, :]
        concat = np.concatenate([x_t, h_last])
        # Reset and Update gates are connected
        r_and_z = sigmoid(concat @ w['kernel'] + w['bias'])
        r, z = r_and_z[:units], r_and_z[units:]

        input_projection = x_t @ w['kernel_h'] + w['bias_h']
        hidden_projection = r * (h_last @ w['recurrent_kernel_h'] + w['recurrent_bias_h'])
        candidate = np.tanh(input_projection + hidden_projection)
        h = (1 - z) * candidate + z * h_last

        yield h.astype(np.float16)
        h_last = h


def numpy_gru_modified(x, w, units):
    """ This is the implementation which supports the kernel,
    recurrent-kernel and bias weights format. """
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    time, features = x.shape
    h_last = np.zeros([units])

    for t in range(time):
        x_t = x[t, :]
        zrh_x = x_t @ w['kernel'] + w['bias'][0]
        z_x = zrh_x[:units]
        r_x = zrh_x[units:units * 2]
        h_x = zrh_x[units * 2:]

        zrh_h = h_last @ w['recurrent_kernel'] + w['bias'][1]
        z_h = zrh_h[:units]
        r_h = zrh_h[units:units * 2]
        h_h = zrh_h[units * 2:]

        r = sigmoid(r_x + r_h)
        z = sigmoid(z_x + z_h)
        candidate = np.tanh(h_x + r * h_h)
        h = (1 - z) * candidate + z * h_last

        yield h.astype(np.float16)
        h_last = h
