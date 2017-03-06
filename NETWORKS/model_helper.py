"""
Model helper for training, saving, restore and evaluation.
"""

import tflearn
from tflearn.layers.core import input_data
from tflearn.layers.estimator import regression
import os


def train_model(network, X, Y, epochs, save_path_file, runId, checkpt_path, tensorboard_dir):
    # Training the model
    print("Training Model...")
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


def load_model(network_builder, model_path, model_file_name, learning_rate, checkpoint_path, tensorboard_dir,
               transf_params_encoded):
    NUM_CLASSES = 10
    input_layer = input_data(shape=[None, 28, 28, 3], name='input')
    softmax = network_builder(input_layer, NUM_CLASSES, transf_params_encoded)
    regress_l = regression(softmax, optimizer='rmsprop',
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


def transfer_model(model, X, Y, epoch, run_id, save_path_file):
    # Restore model training after restoration
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


def layers_to_transfer(code, network_num_layers):  # code = '<num><+/->'
    out = define_layers(network_num_layers)
    for i in range(int(code[:-1])):
        if code[-1] == '+':  # fine tuning
            out[i] = 'FT'
        if code[-1] == '-':  # frozen
            out[i] = 'LK'
    return out


def evaluate_model(model, testX, testY):
    # Evaluate accuracy.
    accuracy_score = model.evaluate(testX, testY, batch_size=32)
    print('Accuracy: %s' % accuracy_score)
    return accuracy_score


def train_model(network, X, Y, epochs, save_path_file, runId, checkpt_path, tensorboard_dir):
    # Training the model
    print("Training model...")
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


def define_layers(n):
    out = []
    for i in range(n - 1):
        out.append('NT')
    out.append('FT')
    return out
