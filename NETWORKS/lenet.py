# coding=utf-8
"""
LeNet-5

Ref.: Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.
Link: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

TFLearn Ref: https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_mnist.py

"""

from __future__ import division, print_function, absolute_import
from .model_helper import transfer_params_decode, define_layers
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

keep_prob = 0.8
# regularizer = "L2"
regularizer = None


# Build LeNet-5
def build_lenet(learning_rate, n_classes=10):
    network = input_data(shape=[None, 28, 28, 3], name='input')

    network = conv_2d(network, 32, 3, activation='relu', regularizer=regularizer, scope='conv1')
    network = max_pool_2d(network, 2, name='maxpool1')
    network = local_response_normalization(network)

    network = conv_2d(network, 64, 3, activation='relu', regularizer=regularizer, scope='conv2')
    network = max_pool_2d(network, 2, name='maxpool2')
    network = local_response_normalization(network)

    network = fully_connected(network, 128, activation='tanh', scope='fc1')
    network = dropout(network, keep_prob, name='dropout1')

    network = fully_connected(network, 256, activation='tanh', scope='fc2')
    network = dropout(network, keep_prob, name='dropout2')

    network = fully_connected(network, n_classes, activation='softmax', scope='fct')

    network = regression(network,
                         optimizer='rmsprop',
                         learning_rate=learning_rate,
                         loss='categorical_crossentropy',
                         name='target')

    return network

# Build LeNet-5 - restore
def lenet(input, num_class, transf_params_encoded=None):
    if transf_params_encoded is None:
        transf_params_encoded = define_layers(5)
    transf_params = transfer_params_decode(transf_params_encoded)

    network = conv_2d(input, 32, 3, activation='relu', regularizer=regularizer, scope='conv1', restore=transf_params[0][0],
                      trainable=transf_params[0][1])
    network = max_pool_2d(network, 2, name='maxpool1')
    network = local_response_normalization(network)

    network = conv_2d(network, 64, 3, activation='relu', regularizer=regularizer, scope='conv2', restore=transf_params[1][0],
                      trainable=transf_params[1][1])
    network = max_pool_2d(network, 2, name='maxpool2')
    network = local_response_normalization(network)

    network = fully_connected(network, 128, activation='tanh', scope='fc1', restore=transf_params[2][0],
                              trainable=transf_params[2][1])
    network = dropout(network, keep_prob, name='dropout1')

    network = fully_connected(network, 256, activation='tanh', scope='fc2', restore=transf_params[3][0],
                              trainable=transf_params[3][1])
    network = dropout(network, keep_prob, name='dropout2')

    network = fully_connected(network, num_class, activation='softmax', scope='fct', restore=transf_params[4][0],
                              trainable=transf_params[4][1])

    return network