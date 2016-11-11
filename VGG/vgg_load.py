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

def vgg11(input, num_class):
    network = conv_2d(input, 8, 3, activation='relu', scope='conv1_1')
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

    network = fully_connected(network, num_class, activation='softmax', scope='fc8', restore=False)

    return network

def vgg16(input, num_class):
    network = conv_2d(input, 8, 3, activation='relu', scope='conv1_1')
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

    network = fully_connected(network, num_class, activation='softmax', scope='fc8', restore=False)

    return network

def load_vgg16(vggnet,model_path,model_file_name,learning_rate,checkpoint_path,tensorboard_dir):
	# VGG Network
	if (vggnet == 11):
		mvggnet = vgg11
	elif (vggnet == 16):
		mvggnet = vgg16
	input_layer = input_data(shape=[None, 28, 28, 3], name='input')
	softmax = mvggnet(input_layer, NUM_CLASSES)
	regression = regression(softmax, optimizer='rmsprop',
	                                loss='categorical_crossentropy',
	                                learning_rate=learning_rate, 
	                                restore=False)
	model = tflearn.DNN(regression, checkpoint_path=checkpoint_path,
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

def evaluate_vgg(model,testX,testY):
	# Evaluate accuracy.
	accuracy_score = model.evaluate(testX,testY,batch_size=32)
	print('Accuracy: %s' %accuracy_score)
	return accuracy_score