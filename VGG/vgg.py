#from tflearn examples: https://github.com/tflearn/tflearn/blob/master/examples/images/vgg_network.py

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
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# Building 'VGG-16 Network'
def build_vgg16(LEARNING_RATE):
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
	network = fully_connected(network, 10, activation='softmax', scope='fc8')

	network = regression(network, 
						optimizer='rmsprop',
						loss='categorical_crossentropy',
						learning_rate=LEARNING_RATE)
	return network

def build_vgg11(LEARNING_RATE):
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
	network = fully_connected(network, 10, activation='softmax', scope='fc8')

	network = regression(network, 
						optimizer='rmsprop',
						loss='categorical_crossentropy',
						learning_rate=LEARNING_RATE)
	return network

def train_vgg(network,X,Y,epochs,save_path_file,runId,checkpt_path,tensorboard_dir):
	# Training the model
	print ("Training VGG...")
	model = tflearn.DNN(network, 
		checkpoint_path=checkpt_path,
		max_checkpoints=3, 
		tensorboard_verbose=0,
		tensorboard_dir=tensorboard_dir)
	model.fit(X, Y, 
		n_epoch=epochs, 
		shuffle=True,
		show_metric=True, 
		batch_size=32, 
		snapshot_epoch=True, 
		run_id=runId,
		validation_set=0.0)
	# Save the model
	model.save(save_path_file)
	print ('Model SAVED!')
	return model