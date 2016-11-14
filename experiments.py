#
#Experiments 
#
import tflearn
import os
import sys
import tensorflow as tf
from VGG.vgg import build_vgg16, build_vgg11, train_vgg
from VGG.model_helper import evaluate_model, train_model
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
def report_line(line):
	appendFile(REPORT_LOG_FILE_NAME,[line])

# Data loading and preprocessing
print ("Data loading and preprocessing...")

X, Y, testX, testY = get_mnist(instances=1000, rgb=True)
# X, Y, testX, testY = get_mnist(rgb=True)


MODEL_PATH = './models/'
CHECKPOINT_PATH = 'checkpoints/'
TENSORBOARDDIR = 'tensorboard/'
checkDir(MODEL_PATH)
checkDir(CHECKPOINT_PATH)
checkDir(TENSORBOARDDIR)

def report(RUN_ID, X,testX,t,EPOCHS, LEARN_RATE, acc):
	strs = []
	strs.append ('ID: '+RUN_ID)
	strs.append ("Dataset in use: train size= %d; test size= %d" %(len(X),len(testX)))
	strs.append ("Training completed in %d s"%(t))
	strs.append ('Epochs: %d'%EPOCHS)
	strs.append ('Learning rate: %f'%LEARN_RATE)
	strs.append (acc)
	strs.append ('\n')
	for i in strs:
		print(i)
		report_line(i)

def train_exec():
	startCrono()

	# Training
	LEARN_RATE = 0.00001
	EPOCHS = 3
	MODEL_NAME = 'vgg11-model1.tfl'
	SAVE_PATH_FILE = MODEL_PATH + MODEL_NAME
	RUN_ID = 'train_vgg11_run_1'
	#
	net = build_vgg11(LEARN_RATE)
	model = train_vgg(net, X, Y,
						EPOCHS,
						SAVE_PATH_FILE,
						RUN_ID,
						CHECKPOINT_PATH,
						TENSORBOARDDIR)
	
	t = getCrono()
	#evaluate model
	acc = evaluate_model(model,testX,testY)
	
	report(RUN_ID, X,testX,t,EPOCHS, LEARN_RATE, acc)

	#run more epochs of training
	startCrono()
	EPOCHS = 2
	RUN_ID = RUN_ID+'_plus_epochs'
	model = train_model(model, X, Y, EPOCHS, RUN_ID, SAVE_PATH_FILE)
	t = getCrono()
	#evaluate model
	acc = evaluate_model(model,testX,testY)
	report(RUN_ID, X,testX,t,EPOCHS, LEARN_RATE, acc)

	return model

def transfer_exec():
	#transfer
	#load model
	startCrono()
	MODEL_NAME = 'vgg11-model1.tfl'
	N_MODEL_NAME = 'vgg11-transfered-model1.tfl'
	SAVE_PATH_FILE = MODEL_PATH + N_MODEL_NAME
	EPOCHS = 5
	LEARN_RATE = 0.00001
	RUN_ID = "transfer_vgg11_run_1"
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
	#evaluate model
	acc = evaluate_model(lmodel,testX,testY)
	report(RUN_ID, X,testX,t,EPOCHS, LEARN_RATE, acc)
	return lmodel

m = train_exec()
tf.reset_default_graph()
n = transfer_exec()
tf.reset_default_graph()

print ('END')
