import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.contrib import learn

from utils import LOG_DIR, flatten_convolution, dense_layer

IMAGE_SIZE = 277

mnist = learn.datasets.load_dataset('mnist')


def alexnet_layer(tensor_in, n_filters, filter_shape, pool_size, activation=tf.nn.tanh,
                  padding='VALID', norm_depth_radius=4, dropout=None):
    conv = learn.ops.conv2d(tensor_in,
                            n_filters=n_filters,
                            filter_shape=filter_shape,
                            activation=activation,
                            padding=padding)
    pool = tf.nn.max_pool(conv, ksize=pool_size, strides=pool_size, padding=padding)
    norm = tf.nn.lrn(pool, depth_radius=norm_depth_radius, alpha=0.001 / 9.0, beta=0.75)
    if dropout:
        norm = learn.ops.dropout(norm, dropout)
    return norm


def alex_conv_pool_layer(tensor_in, n_filters, kernel_size, stride, pool_size, pool_stride,
                         activation_fn=tf.nn.tanh, padding='SAME'):
    conv = tf.contrib.layers.convolution2d(tensor_in,
                                           num_outputs=n_filters,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=activation_fn,
                                           padding=padding)
    pool = tf.nn.max_pool(conv, ksize=pool_size, strides=pool_stride, padding=padding)
    return pool


def alex_3_convs_pool_layer(tensor_in, activation_fn=tf.nn.tanh, padding='SAME'):
    conv = tf.contrib.layers.convolution2d(tensor_in,
                                           num_outputs=384,
                                           kernel_size=[3, 3],
                                           stride=1,
                                           activation_fn=activation_fn,
                                           padding=padding)
    conv = tf.contrib.layers.convolution2d(conv,
                                           num_outputs=384,
                                           kernel_size=[3, 3],
                                           stride=1,
                                           activation_fn=activation_fn,
                                           padding=padding)
    conv = tf.contrib.layers.convolution2d(conv,
                                           num_outputs=256,
                                           kernel_size=[3, 3],
                                           stride=1,
                                           activation_fn=activation_fn,
                                           padding=padding)
    pool = tf.nn.max_pool(conv, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding=padding)
    return pool


def alexnet_model(X, y, image_size=(-1, IMAGE_SIZE, IMAGE_SIZE, 3)):
    y = tf.one_hot(y, 10, 1, 0)
    X = tf.reshape(X, image_size)
    import numpy as np
    X = tf.get_variable('X', initializer=tf.to_float(np.zeros([1, 227, 227, 3])))
    y = tf.get_variable('y', initializer=tf.to_float(np.zeros([1, 1])))

    with tf.variable_scope('layer1'):
        layer1 = alex_conv_pool_layer(X, 96, [11, 11], 4, (1, 3, 3, 1), (1, 2, 2, 1))

    with tf.variable_scope('layer2'):
        layer2 = alex_conv_pool_layer(layer1, 256, [5, 5], 2, (1, 3, 3, 1), (1, 2, 2, 1))

    with tf.variable_scope('layer3'):
        layer3 = alex_3_convs_pool_layer(layer2)
        layer3_flat = flatten_convolution(layer3)

    result = dense_layer(layer3_flat, [4096, 4096, 1000], activation_fn=tf.nn.tanh, keep_prob=0.2)
    prediction, loss = tf.contrib.learn.models.logistic_regression_zero_init(result, y)
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
        learning_rate=0.1)
    return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op


classifier = learn.Estimator(model_fn=alexnet_model, model_dir=LOG_DIR)
classifier.fit(mnist.train.images, mnist.train.labels, steps=10, batch_size=3,
               monitors=[learn.monitors.ValidationMonitor(mnist.validation.images,
                                                          mnist.validation.labels)])
score = metrics.accuracy_score(mnist.test.labels, classifier.predict(mnist.test.images))
print('Accuracy: {0:f}'.format(score))
