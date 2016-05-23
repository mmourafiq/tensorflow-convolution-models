import numpy as np
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.contrib import learn

LOG_DIR = '../ops_logs'
IMAGE_SIZE = 28


def flatten_convolution(tensor_in):
    tendor_in_shape = tensor_in.get_shape()
    tensor_in_flat = tf.reshape(tensor_in, [tendor_in_shape[0].value or -1, np.prod(tendor_in_shape[1:]).value])
    return tensor_in_flat


def lenet_layer(tensor_in, n_filters, filter_shape, pool_size, activation=tf.nn.tanh, padding='SAME'):
    conv = learn.ops.conv2d(tensor_in,
                            n_filters=n_filters,
                            filter_shape=filter_shape,
                            activation=activation,
                            padding=padding)
    pool = tf.nn.max_pool(conv, ksize=pool_size, strides=pool_size, padding=padding)
    return pool


def lenet_model(X, y, image_size=(-1, IMAGE_SIZE, IMAGE_SIZE, 1), pool_size=(1, 2, 2, 1)):
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

    fc = learn.ops.dnn(layer2_flat, [1024], activation=tf.nn.tanh, dropout=0.5)
    return learn.models.logistic_regression(fc, y)


data_file = './data/notMNIST.pickle'

with open(data_file, 'rb') as f:
    import pickle

    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print 'Training set', train_dataset.shape, train_labels.shape
    print 'Validation set', valid_dataset.shape, valid_labels.shape
    print 'Test set', test_dataset.shape, test_labels.shape


def reformat(dataset):
    dataset = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)
    return dataset


train_dataset = reformat(train_dataset)
valid_dataset = reformat(valid_dataset)
test_dataset = reformat(test_dataset)
print 'Training set', train_dataset.shape, train_labels.shape
print 'Validation set', valid_dataset.shape, valid_labels.shape
print 'Test set', test_dataset.shape, test_labels.shape

classifier = learn.TensorFlowEstimator(model_fn=lenet_model, n_classes=10, batch_size=300,
                                       optimizer='Adagrad',
                                       steps=10000, learning_rate=0.001)
classifier.fit(train_dataset, train_labels, logdir=LOG_DIR)
score = metrics.accuracy_score(test_labels, classifier.predict(test_dataset))
print('Accuracy: {0:f}'.format(score))
