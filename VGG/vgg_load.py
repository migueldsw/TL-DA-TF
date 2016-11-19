'''
Retraining (Finetuning) Example with vgg.tflearn. Using weights from VGG-11 or 16 model to retrain
network for a new task.All weights are restored except
last layer (softmax) that will be retrained to match the new task (finetuning).
'''

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import os

def vgg11(input, num_class, transf_params_encoded=['FT','FT','FT','FT','FT','FT','FT','FT','FT','FT','NT']):
	transf_params = transfer_params_decode(transf_params_encoded)

    network = conv_2d(input, 8, 3, activation='relu', scope='conv1_1', restore=transf_params[0][0], trainable=transf_params[0][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool1')

    network = conv_2d(network, 16, 3, activation='relu', scope='conv2_1', restore=transf_params[1][0], trainable=transf_params[1][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool2')

    network = conv_2d(network, 32, 3, activation='relu', scope='conv3_1', restore=transf_params[2][0], trainable=transf_params[2][1])
    network = conv_2d(network, 32, 3, activation='relu', scope='conv3_2', restore=transf_params[3][0], trainable=transf_params[3][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool3')

    network = conv_2d(network, 64, 3, activation='relu', scope='conv4_1', restore=transf_params[4][0], trainable=transf_params[4][1])
    network = conv_2d(network, 64, 3, activation='relu', scope='conv4_2', restore=transf_params[5][0], trainable=transf_params[5][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool4')

    network = conv_2d(network, 64, 3, activation='relu', scope='conv5_1', restore=transf_params[6][0], trainable=transf_params[6][1])
    network = conv_2d(network, 64, 3, activation='relu', scope='conv5_2', restore=transf_params[7][0], trainable=transf_params[7][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool5')

    network = fully_connected(network, 512, activation='relu', scope='fc6', restore=transf_params[8][0], trainable=transf_params[8][1])
    network = dropout(network, 0.5, name='dropout1')

    network = fully_connected(network, 512, activation='relu', scope='fc7', restore=transf_params[9][0], trainable=transf_params[9][1])
    network = dropout(network, 0.5, name='dropout2')

    network = fully_connected(network, num_class, activation='softmax', scope='fc8', restore=transf_params[10][0], trainable=transf_params[10][1])

    return network

def vgg16(input, num_class, ransf_params_encoded=['FT','FT','FT','FT','FT','FT','FT','FT','FT','FT','FT','FT','FT','FT','FT','NT']):
	transf_params = transfer_params_decode(transf_params_encoded)

    network = conv_2d(input, 8, 3, activation='relu', scope='conv1_1', restore=transf_params[0][0], trainable=transf_params[0][1])
    network = conv_2d(network, 8, 3, activation='relu', scope='conv1_2', restore=transf_params[1][0], trainable=transf_params[1][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool1')

    network = conv_2d(network, 16, 3, activation='relu', scope='conv2_1', restore=transf_params[2][0], trainable=transf_params[2][1])
    network = conv_2d(network, 16, 3, activation='relu', scope='conv2_2', restore=transf_params[3][0], trainable=transf_params[3][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool2')

    network = conv_2d(network, 32, 3, activation='relu', scope='conv3_1', restore=transf_params[4][0], trainable=transf_params[4][1])
    network = conv_2d(network, 32, 3, activation='relu', scope='conv3_2', restore=transf_params[5][0], trainable=transf_params[5][1])
    network = conv_2d(network, 32, 3, activation='relu', scope='conv3_3', restore=transf_params[6][0], trainable=transf_params[6][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool3')

    network = conv_2d(network, 64, 3, activation='relu', scope='conv4_1', restore=transf_params[7][0], trainable=transf_params[7][1])
    network = conv_2d(network, 64, 3, activation='relu', scope='conv4_2', restore=transf_params[8][0], trainable=transf_params[8][1])
    network = conv_2d(network, 64, 3, activation='relu', scope='conv4_3', restore=transf_params[9][0], trainable=transf_params[9][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool4')

    network = conv_2d(network, 64, 3, activation='relu', scope='conv5_1', restore=transf_params[10][0], trainable=transf_params[10][1])
    network = conv_2d(network, 64, 3, activation='relu', scope='conv5_2', restore=transf_params[11][0], trainable=transf_params[11][1])
    network = conv_2d(network, 64, 3, activation='relu', scope='conv5_3', restore=transf_params[12][0], trainable=transf_params[12][1])
    network = max_pool_2d(network, 2, strides=2, name='maxpool5')

    network = fully_connected(network, 512, activation='relu', scope='fc6', restore=transf_params[13][0], trainable=transf_params[13][1])
    network = dropout(network, 0.5, name='dropout1')

    network = fully_connected(network, 512, activation='relu', scope='fc7', restore=transf_params[14][0], trainable=transf_params[14][1])
    network = dropout(network, 0.5, name='dropout2')

    network = fully_connected(network, num_class, activation='softmax', scope='fc8', restore=transf_params[15][0], trainable=transf_params[15][1])

    return network

def load_vgg(vggnet,model_path,model_file_name,learning_rate,checkpoint_path,tensorboard_dir):
	NUM_CLASSES = 10
	# VGG Network
	if (vggnet == 11):
		mvggnet = vgg11
	elif (vggnet == 16):
		mvggnet = vgg16
	input_layer = input_data(shape=[None, 28, 28, 3], name='input')
	softmax = mvggnet(input_layer, NUM_CLASSES)
	regress_l = regression(softmax, optimizer='rmsprop',
	                                loss='categorical_crossentropy',
	                                learning_rate=learning_rate, 
	                                restore=False)
	model = tflearn.DNN(regress_l, checkpoint_path=checkpoint_path,
	                    max_checkpoints=3, tensorboard_verbose=2,
	                    tensorboard_dir=tensorboard_dir)
	# Load model weights
	print ('Loading model...')
	model_file = os.path.join(model_path, model_file_name)
	model.load(model_file, weights_only=True)
	print ('Model loaded!')
	return model

def transfer_vgg(model,X,Y,epoch,run_id,save_path_file):
	# Start finetuning
	model.fit(X, Y, n_epoch=epoch, validation_set=0.0, shuffle=True,
	          show_metric=True, batch_size=64, snapshot_epoch=False,
	          snapshot_step=200, run_id=run_id)
	# Save the model
	model.save(save_path_file)
	print ('Model SAVED!')
	return model

def transfer_params_decode(coded_params):
	#returns the params for transfer
	out = []
	for i in coded_params:
		if i == 'FT':
			# (restore, trainable)
			out.append((True, True))
		if i == 'LK':
			# (restore, trainable)
			out.append((True, False))
		if i == 'NT':
			# (restore, trainable)
			out.append((False, True))
	return out