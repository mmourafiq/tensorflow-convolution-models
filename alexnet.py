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


def alexnet_layer(tensor_in, n_filters, filter_shape, pool_size, activation=tf.nn.tanh, padding='VALID',
                  norm_depth_radius=4, dropout=None):
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


def alexnet_model(X, y, image_size=(-1, 28, 28, 1), pool_size=(1, 2, 2, 1), dropout=0.2):
    X = tf.reshape(X, image_size)

    with tf.variable_scope('layer1'):
        layer1 = alexnet_layer(X, 64, [3, 3], pool_size, dropout=dropout)

    with tf.variable_scope('layer2'):
        layer2 = alexnet_layer(layer1, 128, [3, 3], pool_size, dropout=dropout)

    with tf.variable_scope('layer3'):
        layer3 = alexnet_layer(layer2, 256, [3, 3], pool_size, dropout=dropout)
        layer3_flat = flatten_convolution(layer3)

    fc = learn.ops.dnn(layer3_flat, [1024, 1024], activation=tf.nn.relu, dropout=dropout)
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

classifier = learn.TensorFlowEstimator(model_fn=alexnet_model, n_classes=10, batch_size=64,
                                       optimizer='Adagrad',
                                       steps=500000, learning_rate=0.01)
classifier.fit(train_dataset, train_labels, logdir=LOG_DIR)
score = metrics.accuracy_score(test_labels, classifier.predict(test_dataset))
print('Accuracy: {0:f}'.format(score))
