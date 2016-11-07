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
print ("Data loading and preprocessing...")
#
from dataset_helper import get_mnist
# X, Y, testX, testY = get_mnist(instances=10, rgb=True)
X, Y, testX, testY = get_mnist(rgb=True)
#
# from dataset_helper import get_svhn
# X, Y, testX, testY = get_svhn(instances=10, crop=True)
#

# Building 'VGG Network'
network = input_data(shape=[None, 28, 28, 3])

network = conv_2d(network, 8, 3, activation='relu')
network = conv_2d(network, 8, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 16, 3, activation='relu')
network = conv_2d(network, 16, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 32, 3, activation='relu')
network = conv_2d(network, 32, 3, activation='relu')
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')

network = regression(network, optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

#-----------------
#time cost evaluation
from datetime import datetime as dt
TIME = [] #[t_init,t_final]
def startCrono():
	TIME.append(dt.now())
def getCrono(): # returns delta t in seconds
	TIME.append(dt.now())
	deltat = TIME[-1]-TIME[-2]
	return deltat.seconds
#----------------

startCrono()

# Training
print ("Training VGG-16...")
EPOCHS = 50

model = tflearn.DNN(network, checkpoint_path='model_vgg_3',
                    max_checkpoints=1, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=EPOCHS, shuffle=True,
          show_metric=True, batch_size=32, snapshot_step=500,
          snapshot_epoch=True, run_id='vgg_mnist_full_3',validation_set=0.0)

print ("Dataset in use: train size= %d; test size= %d" %(len(X),len(testX)))
print ("Training completed in %d s"%(getCrono()))
print ('Epochs: %d'%EPOCHS)

# Evaluate accuracy.
accuracy_score = model.evaluate(testX,testY,batch_size=32)
print('Accuracy: %s' %accuracy_score)
