#from tflearn examples: https://github.com/tflearn/tflearn/blob/master/examples/images/vgg_network.py

""" Very Deep Convolutional Networks for Large-Scale Visual Recognition.
Applying VGG 16-layers convolutional network to Oxford's 17 Category Flower
Dataset classification task.
References:
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    K. Simonyan, A. Zisserman. arXiv technical report, 2014.
Links:
    http://arxiv.org/pdf/1409.1556
"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
# from dataset_helper import get_small_mnist
# X, Y, testX, testY = get_small_mnist(10000)
#
X = X.reshape(len(X),28,28,1)
testX = testX.reshape(len(testX),28,28,1)
print ("Dataset in use: train size= %d; test size= %d" %(len(X),len(testX)))

# Building 'VGG Network'
network = input_data(shape=[None, 28, 28, 1])

network = conv_2d(network, 8, 1, activation='relu')
network = conv_2d(network, 8, 1, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 16, 1, activation='relu')
network = conv_2d(network, 16, 1, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 32, 1, activation='relu')
network = conv_2d(network, 32, 1, activation='relu')
network = conv_2d(network, 32, 1, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 64, 1, activation='relu')
network = conv_2d(network, 64, 1, activation='relu')
network = conv_2d(network, 64, 1, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 64, 1, activation='relu')
network = conv_2d(network, 64, 1, activation='relu')
network = conv_2d(network, 64, 1, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')

network = regression(network, optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg_2',
                    max_checkpoints=1, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=20, shuffle=True,
          show_metric=True, batch_size=32, snapshot_step=500,
          snapshot_epoch=True, run_id='vgg_mnist_full_2',validation_set=0.1)

# Evaluate accuracy.
accuracy_score = model.evaluate(testX,testY,batch_size=32)
print('Accuracy: %s' %accuracy_score)
