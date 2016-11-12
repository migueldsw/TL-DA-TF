#
#Experiments 
#
import tflearn
import os
import sys
import tensorflow as tf
from VGG.vgg import build_vgg16, build_vgg11, train_vgg, evaluate_vgg
from VGG.vgg_load import load_vgg, transfer_vgg
from DATASETS.dataset_helper import get_mnist, get_svhn
from report import appendFile, checkDir

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

REPORT_LOG_FILE_NAME = 'run.log'
def report(line):
	appendFile(REPORT_LOG_FILE_NAME,[line])

# Data loading and preprocessing
print ("Data loading and preprocessing...")

X, Y, testX, testY = get_mnist(instances=100, rgb=True)
# X, Y, testX, testY = get_mnist(rgb=True)


checkDir('models/')
checkDir('checkpoints/')
checkDir('tensorboard/')
# checkDir('checkpoints/model_vgg11')
# checkDir('tensorboard/vgg11')
# checkDir('checkpoints/model_vgg11_retrain')
# checkDir('tensorboard/vgg11_retrain')

def train_exec():
	startCrono()

	# Training
	LEARN_RATE = 0.00001
	EPOCHS = 2
	MODEL_PATH = './models/'
	MODEL_NAME = 'vgg11-model1.tfl'
	SAVE_PATH_FILE = MODEL_PATH + MODEL_NAME
	RUN_ID = 'train_vgg11_run_1'
	CHECKPOINT_PATH = 'checkpoints/'
	TENSORBOARDDIR = 'tensorboard/'
	#
	net = build_vgg11(LEARN_RATE)
	model = train_vgg(net, X, Y,
						EPOCHS,
						SAVE_PATH_FILE,
						RUN_ID,
						CHECKPOINT_PATH,
						TENSORBOARDDIR)
	t = getCrono()
	print ("Dataset in use: train size= %d; test size= %d" %(len(X),len(testX)))
	print ("Training completed in %d s"%(t))
	print ('Epochs: %d'%EPOCHS)
	report ('ID: '+RUN_ID)
	report ("Dataset in use: train size= %d; test size= %d" %(len(X),len(testX)))
	report ("Training completed in %d s"%(t))
	report ('Epochs: %d'%EPOCHS)
	report ('Learning rate: %f'%LEARN_RATE)
	#evaluate model
	acc = evaluate_vgg(model,testX,testY)
	report(acc)
	report('\n')
	return model

def transfer_exec():
	#transfer
	#load model
	startCrono()
	MODEL_PATH = './models/'
	MODEL_NAME = 'vgg11-model1.tfl'
	N_MODEL_NAME = 'vgg11-transfered-model1.tfl'
	SAVE_PATH_FILE = MODEL_PATH + N_MODEL_NAME
	EPOCHS = 1
	LEARN_RATE = 0.00001
	RUN_ID = "transfer_vgg11_run_1"
	CHECKPOINT_PATH = 'checkpoints/'
	TENSORBOARDDIR = 'tensorboard/'
	lmodel = load_vgg(11,
						MODEL_PATH,
						MODEL_NAME,
						LEARN_RATE,
						CHECKPOINT_PATH,
						TENSORBOARDDIR) 
	lmodel = transfer_vgg(lmodel, X, Y,
							EPOCHS,
							RUN_ID,
							SAVE_PATH_FILE)

	t = getCrono()
	print ("Dataset in use: train size= %d; test size= %d" %(len(X),len(testX)))
	print ("TRANSFER completed in %d s"%(t))
	print ('Epochs: %d'%EPOCHS)
	report ('ID: '+RUN_ID)
	report ("Dataset in use: train size= %d; test size= %d" %(len(X),len(testX)))
	report ("Training completed in %d s"%(t))
	report ('Epochs: %d'%EPOCHS)
	report ('Learning rate: %f'%LEARN_RATE)
	#evaluate model
	acc = evaluate_vgg(lmodel,testX,testY)
	report(acc)
	report('\n')
	print ('END')
	return lmodel

m = train_exec()
tf.reset_default_graph()
n = transfer_exec()
tf.reset_default_graph()
