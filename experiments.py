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

# mX, mY, mtestX, mtestY = get_mnist(instances=50, rgb=True)
# sX, sY, stestX, stestY = get_svhn(instances=50, crop=True)
mX, mY, mtestX, mtestY = get_mnist(rgb=True)
sX, sY, stestX, stestY = get_svhn(crop=True)


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
		sout(i)

def sout(i):
	print(i)
	report_line(i)

def train_exec(X,Y,testX,testY,EPOCHS,MODEL_NAME,RUN_ID):
	startCrono()

	# Training
	LEARN_RATE = 0.00001
	#EPOCHS = 3
	# MODEL_NAME = 'vgg11-model1.tfl'
	SAVE_PATH_FILE = MODEL_PATH + MODEL_NAME
	# RUN_ID = 'train_vgg11_run_1'
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

	# #run more epochs of training
	# startCrono()
	# EPOCHS = 2
	# #RUN_ID = RUN_ID+'_plus_epochs'
	# model = train_model(model, X, Y, EPOCHS, RUN_ID, SAVE_PATH_FILE)
	# t = getCrono()
	# #evaluate model
	# acc = evaluate_model(model,testX,testY)
	# report(RUN_ID, X,testX,t,EPOCHS, LEARN_RATE, acc)
	tf.reset_default_graph()
	#return model

def transfer_exec(X,Y,testX,testY,EPOCHS,MODEL_NAME,RUN_ID, N_MODEL_NAME):
	#transfer
	#load model
	startCrono()
	#MODEL_NAME = 'vgg11-model1.tfl'
	#N_MODEL_NAME = 'vgg11-transfered-model1.tfl'
	SAVE_PATH_FILE = MODEL_PATH + N_MODEL_NAME
	#EPOCHS = 5
	LEARN_RATE = 0.00001
	#RUN_ID = "transfer_vgg11_run_1"
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
	tf.reset_default_graph()
	#return lmodel


# DA
train_exec(mX,mY,mtestX,mtestY,20,'model_A.tfl',"train_A")

# DB
train_exec(sX,sY,stestX,stestY,20,'model_B.tfl',"train_B")

# DA|DA
train_exec(mX,mY,mtestX,mtestY,10,'model_A1.tfl',"train_A1")
transfer_exec(mX,mY,mtestX,mtestY,10,'model_A1.tfl',"transf_AA", 'model_AA.tfl')

# DB|DB
train_exec(sX,sY,stestX,stestY,10,'model_B1.tfl',"train_B1")
transfer_exec(sX,sY,stestX,stestY,10,'model_B1.tfl',"transf_BB", 'model_BB.tfl')

# DA|DB
train_exec(mX,mY,mtestX,mtestY,10,'model_A2.tfl',"train_A2")
transfer_exec(sX,sY,stestX,stestY,10,'model_A2.tfl',"transf_AB", 'model_AB.tfl')

# DB|DA
train_exec(sX,sY,stestX,stestY,10,'model_B2.tfl',"train_B2")
transfer_exec(mX,mY,mtestX,mtestY,10,'model_B2.tfl',"transf_BA", 'model_BA.tfl')


sout('END!')
