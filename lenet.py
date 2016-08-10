import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.contrib import learn

from utils import LOG_DIR, dense_layer, flatten_convolution

IMAGE_SIZE = 28
mnist = learn.datasets.load_dataset('mnist')


def lenet_layer(tensor_in, n_filters, kernel_size, pool_size, activation_fn=tf.nn.tanh,
                padding='SAME'):
    conv = tf.contrib.layers.convolution2d(tensor_in,
                                           num_outputs=n_filters,
                                           kernel_size=kernel_size,
                                           activation_fn=activation_fn,
                                           padding=padding)
    pool = tf.nn.max_pool(conv, ksize=pool_size, strides=pool_size, padding=padding)
    return pool


def lenet_model(X, y, image_size=(-1, IMAGE_SIZE, IMAGE_SIZE, 1), pool_size=(1, 2, 2, 1)):
    y = tf.one_hot(y, 10, 1, 0)
    X = tf.reshape(X, image_size)

    with tf.variable_scope('layer1'):
        """
        Valid:
         * input: (?, 28, 28, 1)
         * filter: (5, 5, 1, 4)
         * pool: (1, 2, 2, 1)
         * output: (?, 12, 12, 4)
        Same:
         * input: (?, 28, 28, 1)
         * filter: (5, 5, 1, 4)
         * pool: (1, 2, 2, 1)
         * output: (?, 14, 14, 4)
        """
        layer1 = lenet_layer(X, 4, [5, 5], pool_size)

    with tf.variable_scope('layer2'):
        """
        VALID:
         * input: (?, 12, 12, 4)
         * filter: (5, 5, 4, 6)
         * pool: (1, 2, 2, 1)
         * output: (?, 4, 4, 6)
         * flat_output: (?, 4 * 4 * 6)
        SAME:
         * input: (?, 14, 14, 4)
         * filter: (5, 5, 4, 6)
         * pool: (1, 2, 2, 1)
         * output: (?, 7, 7, 6)
         * flat_output: (?, 7 * 7 * 6)
        """
        layer2 = lenet_layer(layer1, 6, [5, 5], pool_size)
        layer2_flat = flatten_convolution(layer2)

    result = dense_layer(layer2_flat, [1024], activation_fn=tf.nn.tanh, keep_prob=0.5)
    prediction, loss = tf.contrib.learn.models.logistic_regression_zero_init(result, y)
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
        learning_rate=0.1)
    return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op


classifier = learn.Estimator(model_fn=lenet_model, model_dir=LOG_DIR)
classifier.fit(mnist.train.images, mnist.train.labels, steps=10000, batch_size=300,
               monitors=[learn.monitors.ValidationMonitor(mnist.validation.images,
                                                          mnist.validation.labels)])
score = metrics.accuracy_score(mnist.test.labels, classifier.predict(mnist.test.images))
print('Accuracy: {0:f}'.format(score))
