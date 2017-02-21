""" AlexNet
Dataset classification task: 28x28x3 shape (adjusted AlexNet)
References:
	ImageNet Classification with Deep Convolutional Neural Networks.
	K. Simonyan, A. Zisserman. arXiv technical report, 2014.
Links:
	http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""

from __future__ import division, print_function, absolute_import
import sys
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

# keep_prob = float(sys.argv[4]) # keep_prob \in (0, 1]
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
    network = conv_2d(network, 48, 3, activation='relu', scope='3_1')
    network = conv_2d(network, 48, 3, activation='relu', scope='3_2')
    network = conv_2d(network, 32, 3, activation='relu', scope='3_3')
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


def train_alexnet(network, X, Y, epochs, save_path_file, runId, checkpt_path, tensorboard_dir):
    # Training the model
    print("Training AlexNet...")
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
    print('Model SAVED!')
    return model
