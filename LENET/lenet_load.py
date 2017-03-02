"""
Retraining (Finetuning) using weights from LeNet model to retrain
network for a new task.All weights are restored except
last layer (softmax) that will be retrained to match the new task (finetuning).
"""

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
import os

keep_prob = 0.8

#Build LeNet
def lenet(input, num_class, transf_params_encoded=['FT', 'FT', 'FT', 'FT', 'NT']):
    transf_params = transfer_params_decode(transf_params_encoded)

    network = conv_2d(input, 32, 3, activation='relu', regularizer="L2", scope='conv1', restore=transf_params[0][0],
                      trainable=transf_params[0][1])
    network = max_pool_2d(network, 2, name='maxpool1')
    network = local_response_normalization(network)

    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2", scope='conv2', restore=transf_params[1][0],
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

def load_lenet(model_path, model_file_name, learning_rate, checkpoint_path, tensorboard_dir,
                 transf_params_encoded):
    NUM_CLASSES = 10
    # LeNet Network
    input_layer = input_data(shape=[None, 28, 28, 3], name='input')
    softmax = lenet(input_layer, NUM_CLASSES, transf_params_encoded)
    regress_l = regression(softmax, optimizer='adam',
                           loss='categorical_crossentropy',
                           learning_rate=learning_rate,
                           restore=False)
    model = tflearn.DNN(regress_l, checkpoint_path=checkpoint_path,
                        max_checkpoints=3, tensorboard_verbose=0,
                        tensorboard_dir=tensorboard_dir)
    # Load model weights
    print ('Loading model...')
    model_file = os.path.join(model_path, model_file_name)
    model.load(model_file, weights_only=True)
    print ('Model loaded!')
    return model

def transfer_lenet(model, X, Y, epoch, run_id, save_path_file):
    # Start finetuning
    model.fit(X, Y, n_epoch=epoch, validation_set=0.0, shuffle=True,
              show_metric=True, batch_size=64, snapshot_epoch=True,
              run_id=run_id)
    # Save the model
    model.save(save_path_file)
    print ('Model SAVED!')
    return model


def transfer_params_decode(coded_params):
    # returns the params for transfer
    out = []
    for i in coded_params:
        if i == 'FT':  # Fine Tuning or Unlocked
            # (restore, trainable)
            out.append((True, True))
        if i == 'LK':  # Locked or Frozen
            # (restore, trainable)
            out.append((True, False))
        if i == 'NT':  # Normal Train
            # (restore, trainable)
            out.append((False, True))
    return out