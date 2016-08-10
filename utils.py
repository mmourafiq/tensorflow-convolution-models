import numpy as np
import tensorflow as tf

LOG_DIR = './ops_logs'


def flatten_convolution(tensor_in):
    tendor_in_shape = tensor_in.get_shape()
    tensor_in_flat = tf.reshape(tensor_in, [tendor_in_shape[0].value or -1, np.prod(tendor_in_shape[1:]).value])
    return tensor_in_flat


def dense_layer(tensor_in, layers, activation_fn=tf.nn.tanh, keep_prob=None):
    if not keep_prob:
        return tf.contrib.layers.stack(
            tensor_in, tf.contrib.layers.fully_connected, layers, activation_fn=activation_fn)

    tensor_out = tensor_in
    for layer in layers:
        tensor_out = tf.contrib.layers.fully_connected(tensor_out, layer,
                                                       activation_fn=activation_fn)
        tensor_out = tf.contrib.layers.dropout(tensor_out, keep_prob=keep_prob)

    return tensor_out
