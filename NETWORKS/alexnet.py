""" AlexNet
Dataset classification task: 28x28x3 shape (adjusted AlexNet)
References:
	ImageNet Classification with Deep Convolutional Neural Networks.
	K. Simonyan, A. Zisserman. arXiv technical report, 2014.
Links:
	http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""

from __future__ import division, print_function, absolute_import
from .model_helper import transfer_params_decode, define_layers
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

keep_prob = 0.5

# Building 'AlexNet Network'
def build_alexnet(learning_rate, n_class=10):
    network = input_data(shape=[None, 28, 28, 3], name='input')

    network = conv_2d(network, 12, 11, strides=4, activation='relu', scope='conv1')
    network = max_pool_2d(network, 3, strides=2, name='maxpool1')
    network = local_response_normalization(network)
    network = conv_2d(network, 32, 5, activation='relu', scope='conv2')
    network = max_pool_2d(network, 3, strides=2, name='maxpool2')
    network = local_response_normalization(network)
    network = conv_2d(network, 48, 3, activation='relu', scope='conv3_1')
    network = conv_2d(network, 48, 3, activation='relu', scope='conv3_2')
    network = conv_2d(network, 32, 3, activation='relu', scope='conv3_3')
    network = max_pool_2d(network, 3, strides=2, name='maxpool3')
    network = local_response_normalization(network)
    network = fully_connected(network, 512, activation='tanh', scope='fc1')
    network = dropout(network, keep_prob, name='dropout1')
    network = fully_connected(network, 512, activation='tanh', scope='fc2')
    network = dropout(network, keep_prob, name='dropout2')
    network = fully_connected(network, n_class, activation='softmax', scope='fc3')

    network = regression(network,
                         optimizer='rmsprop',
                         loss='categorical_crossentropy',
                         learning_rate=learning_rate)
    return network

def alexnet(input, num_class, transf_params_encoded=None):
    if transf_params_encoded is None:
        transf_params_encoded = define_layers(8)
    transf_params = transfer_params_decode(transf_params_encoded)

    network = conv_2d(input, 12, 11, strides=4, activation='relu', scope='conv1', restore=transf_params[0][0],
                      trainable=transf_params[0][1])
    network = max_pool_2d(network, 3, strides=2, name='maxpool1')
    network = local_response_normalization(network)
    network = conv_2d(network, 32, 5, activation='relu', scope='conv2', restore=transf_params[1][0],
                      trainable=transf_params[1][1])
    network = max_pool_2d(network, 3, strides=2, name='maxpool2')
    network = local_response_normalization(network)
    network = conv_2d(network, 48, 3, activation='relu', scope='conv3_1', restore=transf_params[2][0],
                      trainable=transf_params[2][1])
    network = conv_2d(network, 48, 3, activation='relu', scope='conv3_2', restore=transf_params[3][0],
                      trainable=transf_params[3][1])
    network = conv_2d(network, 32, 3, activation='relu', scope='conv3_3', restore=transf_params[4][0],
                      trainable=transf_params[4][1])
    network = max_pool_2d(network, 3, strides=2, name='maxpool3')
    network = local_response_normalization(network)
    network = fully_connected(network, 512, activation='tanh', scope='fc1', restore=transf_params[5][0],
                              trainable=transf_params[5][1])
    network = dropout(network, keep_prob, name='dropout1')
    network = fully_connected(network, 512, activation='tanh', scope='fc2', restore=transf_params[6][0],
                              trainable=transf_params[6][1])
    network = dropout(network, keep_prob, name='dropout2')
    network = fully_connected(network, num_class, activation='softmax', scope='fc3', restore=transf_params[7][0],
                              trainable=transf_params[7][1])
    return network