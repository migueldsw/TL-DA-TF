# from tflearn examples: https://github.com/tflearn/tflearn/blob/master/examples/images/vgg_network.py

""" Very Deep Convolutional Networks for Large-Scale Visual Recognition.
Applying VGG 11 and 16-layers convolutional network to MNIST and SVHN
Dataset classification task. 28x28x3 shape
References:
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    K. Simonyan, A. Zisserman. arXiv technical report, 2014.
Links:
    http://arxiv.org/pdf/1409.1556
"""

from __future__ import division, print_function, absolute_import
from .model_helper import transfer_params_decode, define_layers
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression


# Building 'VGG-16 Network'
def build_vgg16(learning_rate,num_class=10):
    network = input_data(shape=[None, 28, 28, 3], name='input')

    network = conv_2d(network, 8, 3, activation='relu', scope='conv1_1')
    network = conv_2d(network, 8, 3, activation='relu', scope='conv1_2')
    network = max_pool_2d(network, 2, strides=2, name='maxpool1')

    network = conv_2d(network, 16, 3, activation='relu', scope='conv2_1')
    network = conv_2d(network, 16, 3, activation='relu', scope='conv2_2')
    network = max_pool_2d(network, 2, strides=2, name='maxpool2')

    network = conv_2d(network, 32, 3, activation='relu', scope='conv3_1')
    network = conv_2d(network, 32, 3, activation='relu', scope='conv3_2')
    network = conv_2d(network, 32, 3, activation='relu', scope='conv3_3')
    network = max_pool_2d(network, 2, strides=2, name='maxpool3')

    network = conv_2d(network, 64, 3, activation='relu', scope='conv4_1')
    network = conv_2d(network, 64, 3, activation='relu', scope='conv4_2')
    network = conv_2d(network, 64, 3, activation='relu', scope='conv4_3')
    network = max_pool_2d(network, 2, strides=2, name='maxpool4')

    network = conv_2d(network, 64, 3, activation='relu', scope='conv5_1')
    network = conv_2d(network, 64, 3, activation='relu', scope='conv5_2')
    network = conv_2d(network, 64, 3, activation='relu', scope='conv5_3')
    network = max_pool_2d(network, 2, strides=2, name='maxpool5')

    network = fully_connected(network, 512, activation='relu', scope='fc6')
    network = dropout(network, 0.5, name='dropout1')
    network = fully_connected(network, 512, activation='relu', scope='fc7')
    network = dropout(network, 0.5, name='dropout2')
    network = fully_connected(network, num_class, activation='softmax', scope='fc8')

    network = regression(network,
                         optimizer='rmsprop',
                         loss='categorical_crossentropy',
                         learning_rate=learning_rate)
    return network


# Building 'VGG-11 Network'
def build_vgg11(learning_rate, num_class=10):
    network = input_data(shape=[None, 28, 28, 3], name='input')

    network = conv_2d(network, 8, 3, activation='relu', scope='conv1_1')
    network = max_pool_2d(network, 2, strides=2, name='maxpool1')

    network = conv_2d(network, 16, 3, activation='relu', scope='conv2_1')
    network = max_pool_2d(network, 2, strides=2, name='maxpool2')

    network = conv_2d(network, 32, 3, activation='relu', scope='conv3_1')
    network = conv_2d(network, 32, 3, activation='relu', scope='conv3_2')
    network = max_pool_2d(network, 2, strides=2, name='maxpool3')

    network = conv_2d(network, 64, 3, activation='relu', scope='conv4_1')
    network = conv_2d(network, 64, 3, activation='relu', scope='conv4_2')
    network = max_pool_2d(network, 2, strides=2, name='maxpool4')

    network = conv_2d(network, 64, 3, activation='relu', scope='conv5_1')
    network = conv_2d(network, 64, 3, activation='relu', scope='conv5_2')
    network = max_pool_2d(network, 2, strides=2, name='maxpool5')

    network = fully_connected(network, 512, activation='relu', scope='fc6')
    network = dropout(network, 0.5, name='dropout1')
    network = fully_connected(network, 512, activation='relu', scope='fc7')
    network = dropout(network, 0.5, name='dropout2')
    network = fully_connected(network, num_class, activation='softmax', scope='fc8')

    network = regression(network,
                         optimizer='rmsprop',
                         loss='categorical_crossentropy',
                         learning_rate=learning_rate)
    return network


#
# Retraining (Finetuning) Example with vgg.tflearn. Using weights from VGG-11 or 16 model to retrain
# network for a new task.All weights are restored except
# last layer (softmax) that will be retrained to match the new task (finetuning).
# "

def vgg11(input, num_class, transf_params_encoded=None):
    if transf_params_encoded is None:
        transf_params_encoded = define_layers(11)
    transf_params = transfer_params_decode(transf_params_encoded)

    network = conv_2d(input, 8, 3, activation='relu', scope='conv1_1', restore=transf_params[0][0],
                      trainable=transf_params[0][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool1')

    network = conv_2d(network, 16, 3, activation='relu', scope='conv2_1', restore=transf_params[1][0],
                      trainable=transf_params[1][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool2')

    network = conv_2d(network, 32, 3, activation='relu', scope='conv3_1', restore=transf_params[2][0],
                      trainable=transf_params[2][1])
    network = conv_2d(network, 32, 3, activation='relu', scope='conv3_2', restore=transf_params[3][0],
                      trainable=transf_params[3][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool3')

    network = conv_2d(network, 64, 3, activation='relu', scope='conv4_1', restore=transf_params[4][0],
                      trainable=transf_params[4][1])
    network = conv_2d(network, 64, 3, activation='relu', scope='conv4_2', restore=transf_params[5][0],
                      trainable=transf_params[5][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool4')

    network = conv_2d(network, 64, 3, activation='relu', scope='conv5_1', restore=transf_params[6][0],
                      trainable=transf_params[6][1])
    network = conv_2d(network, 64, 3, activation='relu', scope='conv5_2', restore=transf_params[7][0],
                      trainable=transf_params[7][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool5')

    network = fully_connected(network, 512, activation='relu', scope='fc6', restore=transf_params[8][0],
                              trainable=transf_params[8][1])
    network = dropout(network, 0.5, name='dropout1')

    network = fully_connected(network, 512, activation='relu', scope='fc7', restore=transf_params[9][0],
                              trainable=transf_params[9][1])
    network = dropout(network, 0.5, name='dropout2')

    network = fully_connected(network, num_class, activation='softmax', scope='fc8', restore=transf_params[10][0],
                              trainable=transf_params[10][1])

    return network


def vgg16(input, num_class,
          transf_params_encoded=None):
    if transf_params_encoded is None:
        transf_params_encoded = define_layers(16)
    transf_params = transfer_params_decode(transf_params_encoded)

    network = conv_2d(input, 8, 3, activation='relu', scope='conv1_1', restore=transf_params[0][0],
                      trainable=transf_params[0][1])
    network = conv_2d(network, 8, 3, activation='relu', scope='conv1_2', restore=transf_params[1][0],
                      trainable=transf_params[1][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool1')

    network = conv_2d(network, 16, 3, activation='relu', scope='conv2_1', restore=transf_params[2][0],
                      trainable=transf_params[2][1])
    network = conv_2d(network, 16, 3, activation='relu', scope='conv2_2', restore=transf_params[3][0],
                      trainable=transf_params[3][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool2')

    network = conv_2d(network, 32, 3, activation='relu', scope='conv3_1', restore=transf_params[4][0],
                      trainable=transf_params[4][1])
    network = conv_2d(network, 32, 3, activation='relu', scope='conv3_2', restore=transf_params[5][0],
                      trainable=transf_params[5][1])
    network = conv_2d(network, 32, 3, activation='relu', scope='conv3_3', restore=transf_params[6][0],
                      trainable=transf_params[6][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool3')

    network = conv_2d(network, 64, 3, activation='relu', scope='conv4_1', restore=transf_params[7][0],
                      trainable=transf_params[7][1])
    network = conv_2d(network, 64, 3, activation='relu', scope='conv4_2', restore=transf_params[8][0],
                      trainable=transf_params[8][1])
    network = conv_2d(network, 64, 3, activation='relu', scope='conv4_3', restore=transf_params[9][0],
                      trainable=transf_params[9][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool4')

    network = conv_2d(network, 64, 3, activation='relu', scope='conv5_1', restore=transf_params[10][0],
                      trainable=transf_params[10][1])
    network = conv_2d(network, 64, 3, activation='relu', scope='conv5_2', restore=transf_params[11][0],
                      trainable=transf_params[11][1])
    network = conv_2d(network, 64, 3, activation='relu', scope='conv5_3', restore=transf_params[12][0],
                      trainable=transf_params[12][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool5')

    network = fully_connected(network, 512, activation='relu', scope='fc6', restore=transf_params[13][0],
                              trainable=transf_params[13][1])
    network = dropout(network, 0.5, name='dropout1')

    network = fully_connected(network, 512, activation='relu', scope='fc7', restore=transf_params[14][0],
                              trainable=transf_params[14][1])
    network = dropout(network, 0.5, name='dropout2')

    network = fully_connected(network, num_class, activation='softmax', scope='fc8', restore=transf_params[15][0],
                              trainable=transf_params[15][1])

    return network
