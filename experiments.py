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
def checkDir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
#----------------


# Data loading and preprocessing
print ("Data loading and preprocessing...")

X, Y, testX, testY = get_mnist(instances=100, rgb=True)


checkDir('./models/')
checkDir('checkpoints/model_vgg11')
checkDir('tensorboard/vgg11')
checkDir('checkpoints/model_vgg11_retrain')
checkDir('tensorboard/vgg11_retrain')

def train_exec():
	startCrono()

	# Training
	LEARN_RATE = 0.0001
	EPOCHS = 2
	SAVE_PATH_FILE = './models/vgg11-model1.tfl'
	RUN_ID = 'RUN_1_VGG11'
	CHECKPOINT_PATH = 'checkpoints/model_vgg11'
	TENSORBOARDDIR = 'tensorboard/vgg11'
	#
	net = build_vgg11(LEARN_RATE)
	model = train_vgg(net, X, Y,
						EPOCHS,
						SAVE_PATH_FILE,
						RUN_ID,
						CHECKPOINT_PATH,
						TENSORBOARDDIR)
	print ("Dataset in use: train size= %d; test size= %d" %(len(X),len(testX)))
	print ("Training completed in %d s"%(getCrono()))
	print ('Epochs: %d'%EPOCHS)
	#evaluate model
	acc1 = evaluate_vgg(model,testX,testY)
	return model

def transfer_exec():
	#transfer
	#load model
	startCrono()
	MODEL_PATH = './models/'
	MODEL_NAME = 'vgg11-model1.tfl'
	N_EPOCHS = 1
	LEARN_RATE = 0.0001
	RUN_ID = "run_id1"
	N_SAVE_PATH_FILE = './models/vgg11-transfered-model1.tfl'
	CHECKPOINT_PATH = 'checkpoints/model_vgg11_retrain'
	TENSORBOARDDIR = 'tensorboard/vgg11_retrain'
	lmodel = load_vgg(11,
						MODEL_PATH,
						MODEL_NAME,
						LEARN_RATE,
						CHECKPOINT_PATH,
						TENSORBOARDDIR) 
	lmodel = transfer_vgg(lmodel, X, Y,
							N_EPOCHS,
							RUN_ID,
							N_SAVE_PATH_FILE)

	print ("Dataset in use: train size= %d; test size= %d" %(len(X),len(testX)))
	print ("TRANSFER completed in %d s"%(getCrono()))
	print ('Epochs: %d'%N_EPOCHS)
	#evaluate model
	acc2 = evaluate_vgg(lmodel,testX,testY)
	print ('END')
	return lmodel

m = train_exec()
tf.reset_default_graph()
n = transfer_exec()
tf.reset_default_graph()
