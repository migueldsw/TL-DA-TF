"""
LeNet

Ref.: Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.
Link: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

TFLearn Ref: https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_mnist.py

"""

from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

keep_prob = 0.8

#Build LeNet
def build_lenet(learning_rate, n_classes = 10):
    network = input_data(shape=[None, 28, 28, 3], name='input')

    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2", scope='conv1')
    network = max_pool_2d(network, 2, name='maxpool1')
    network = local_response_normalization(network)

    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2", scope='conv2')
    network = max_pool_2d(network, 2, name='maxpool2')
    network = local_response_normalization(network)

    network = fully_connected(network, 128, activation='tanh', scope='fc1')
    network = dropout(network, keep_prob, name='dropout1')

    network = fully_connected(network, 256, activation='tanh', scope='fc2')
    network = dropout(network, keep_prob, name='dropout2')

    network = fully_connected(network, n_classes, activation='softmax', scope='fct')

    network = regression(network, optimizer='adam', learning_rate=learning_rate,
                         loss='categorical_crossentropy', name='target')

    return network

def train_lenet(network, X, Y, epochs, save_path_file, runId, checkpt_path, tensorboard_dir):
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