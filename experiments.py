#
# Experiments
#
import tflearn
import os
import sys
import tensorflow as tf
from VGG.vgg import build_vgg16, build_vgg11, train_vgg
from VGG.model_helper import evaluate_model, train_model, layers_to_transfer
from VGG.vgg_load import load_vgg, transfer_vgg
from LENET.lenet import build_lenet
from LENET.lenet_load import load_lenet, transfer_lenet
from DATASETS.dataset_helper import get_mnist, get_svhn, get_cifar10
from report import appendFile, checkDir

# UTILS------------------------------------------
# time cost evaluation
from datetime import datetime as dt

TIME = []  # [t_init,t_final]


def startCrono():
    TIME.append(dt.now())


def getCrono():  # returns delta t in seconds
    TIME.append(dt.now())
    deltat = TIME[-1] - TIME[-2]
    return deltat.seconds


# report file writer
def report_line(line):
    appendFile(REPORT_LOG_FILE_NAME, [line])


REPORT_LOG_FILE_NAME = 'run.log'
# -------------------------------------------------


## GLOBAL VALUES -----------------------------------------------
# Datasets loading and preprocessing
print ("Data loading and preprocessing...")
mX, mY, mtestX, mtestY = get_mnist(instances=50, rgb=True)
sX, sY, stestX, stestY = mX, mY, mtestX, mtestY
cX, cY, ctestX, ctestY = mX, mY, mtestX, mtestY
# sX, sY, stestX, stestY = get_svhn(instances=50, crop=True)
# cX, cY, ctestX, ctestY = get_cifar10(instances=50, crop=True)
# mX, mY, mtestX, mtestY = get_mnist(rgb=True)
# sX, sY, stestX, stestY = get_svhn(crop=True)
# cX, cY, ctestX, ctestY = get_cifar10(crop=True)

EPOCHS = 1
# EPOCHS = 50
LEARN_RATE = 0.00001

MODEL_PATH = './models/'
CHECKPOINT_PATH = 'checkpoints/'
TENSORBOARDDIR = 'tensorboard/'
checkDir(MODEL_PATH)
checkDir(CHECKPOINT_PATH)
checkDir(TENSORBOARDDIR)


# --------------------------------------------------------------

def report(RUN_ID, X, testX, t, EPOCHS, LEARN_RATE, acc):
    strs = []
    strs.append('ID: ' + RUN_ID)
    strs.append("Dataset in use: train size= %d; test size= %d" % (len(X), len(testX)))
    strs.append("Training completed in %d s" % (t))
    strs.append('Epochs: %d' % EPOCHS)
    strs.append('Learning rate: %f' % LEARN_RATE)
    strs.append(acc)
    strs.append('\n')
    for i in strs:
        sout(i)


def sout(i):
    print(i)
    report_line(i)


def train_exec(build_vgg, X, Y, testX, testY, MODEL_NAME, RUN_ID):
    startCrono()
    # Training
    SAVE_PATH_FILE = MODEL_PATH + MODEL_NAME
    net = build_vgg(LEARN_RATE)
    model = train_vgg(net, X, Y,
                      EPOCHS,
                      SAVE_PATH_FILE,
                      RUN_ID,
                      CHECKPOINT_PATH,
                      TENSORBOARDDIR)
    t = getCrono()
    # evaluate model
    acc = evaluate_model(model, testX, testY)
    report(RUN_ID, X, testX, t, EPOCHS, LEARN_RATE, acc)
    tf.reset_default_graph()


# return model

def transfer_exec(vgg, X, Y, testX, testY, MODEL_NAME, RUN_ID, N_MODEL_NAME, transf_params_encoded):
    # transfer
    # load model
    startCrono()
    SAVE_PATH_FILE = MODEL_PATH + N_MODEL_NAME
    lmodel = load_vgg(vgg,
                      MODEL_PATH,
                      MODEL_NAME,
                      LEARN_RATE,
                      CHECKPOINT_PATH,
                      TENSORBOARDDIR,
                      transf_params_encoded)
    lmodel = transfer_vgg(lmodel, X, Y,
                          EPOCHS / 2,
                          RUN_ID,
                          SAVE_PATH_FILE)
    t = getCrono()
    # evaluate model
    acc = evaluate_model(lmodel, testX, testY)
    report(RUN_ID, X, testX, t, EPOCHS / 2, LEARN_RATE, acc)
    tf.reset_default_graph()


# return lmodel

def pretrain_transfer(dataset, build_vgg):
    MODEL_NAME = 'model_' + dataset + '.tfl'
    RUN_ID = 'pretrain_' + dataset
    if (dataset == 'A'):
        X, Y, testX, testY = mX, mY, mtestX, mtestY
    elif (dataset == 'B'):
        X, Y, testX, testY = sX, sY, stestX, stestY
    elif (dataset == 'C'):
        X, Y, testX, testY = cX, cY, ctestX, ctestY
    startCrono()
    # Training
    SAVE_PATH_FILE = MODEL_PATH + MODEL_NAME
    net = build_vgg(LEARN_RATE)
    model = train_vgg(net, X, Y,
                      EPOCHS / 2,
                      SAVE_PATH_FILE,
                      RUN_ID,
                      CHECKPOINT_PATH,
                      TENSORBOARDDIR)
    t = getCrono()
    # evaluate model
    acc = evaluate_model(model, testX, testY)
    report(RUN_ID, X, testX, t, EPOCHS / 2, LEARN_RATE, acc)
    tf.reset_default_graph()


# return model

def transfer(params_str, vgg):  # Dn_D
    sourceDataset = params_str[0]
    targetDataset = params_str[-1]
    transferParams = params_str[1:-1]

    if (targetDataset == 'A'):
        X, Y, testX, testY = mX, mY, mtestX, mtestY
    elif (targetDataset == 'B'):
        X, Y, testX, testY = sX, sY, stestX, stestY
    elif (targetDataset == 'C'):
        X, Y, testX, testY = cX, cY, ctestX, ctestY

    trained_model_name = 'model_' + sourceDataset + '.tfl'
    transfer_exec(vgg, X, Y, testX, testY, trained_model_name, params_str, params_str + '.tfl',
                  layers_to_transfer(transferParams, vgg))


def EXEC_BASE(n):
    n = str(n)
    train_exec(build_vgg11, mX, mY, mtestX, mtestY, 'VGG11_A' + n + '.tfl', "train11_A" + n)
    train_exec(build_vgg16, mX, mY, mtestX, mtestY, 'VGG16_A' + n + '.tfl', "train16_A" + n)
    train_exec(build_vgg11, sX, sY, stestX, stestY, 'VGG11_B' + n + '.tfl', "train11_B" + n)
    train_exec(build_vgg16, sX, sY, stestX, stestY, 'VGG16_B' + n + '.tfl', "train16_B" + n)
    train_exec(build_vgg11, cX, cY, ctestX, ctestY, 'VGG11_C' + n + '.tfl', "train11_C" + n)
    train_exec(build_vgg16, cX, cY, ctestX, ctestY, 'VGG16_C' + n + '.tfl', "train16_C" + n)


def EXEC_TRANSFER():
    build_vgg = build_vgg11
    vgg = 11
    for D in ['A', 'B', 'C']:
        pretrain_transfer(D, build_vgg)
    for S in ['A', 'B', 'C']:
        for T in ['A', 'B', 'C']:
            for layer in ['1', '2', '4', '6', '8', '10']:
                for mode in ['+', '-']:
                    transfer(S + layer + mode + T, vgg)


from subprocess import call

BASE_EXPERIMENTS = 0
for n in range(BASE_EXPERIMENTS):
    EXEC_BASE(str(n))
    call(["sh", "extras/backup.sh", "BKP-BASE-" + str(n)])

TRANSFER_EXPERIMENTS = 0
for n in range(TRANSFER_EXPERIMENTS):
    EXEC_TRANSFER()
    call(["sh", "extras/backup.sh", "BKP-TRANSFER-" + str(n)])


def exec_lenet(n):
    n = str(n)
    train_exec(build_lenet, mX, mY, mtestX, mtestY, 'LENET_A' + n + '.tfl', "trainLeNet_A" + n)


def run_exp(model, epoch, exps):
    print "Experiments with:"
    print "Model", model
    print "Epochs = ", epoch
    print "Experiments = ", exps
    EPOCHS = int(epoch)
    for i in range(int(exps)):
        exec_lenet(i)
    sout('END!')

print "Loaded!"


