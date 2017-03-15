#
# Experiments helper
#

import tensorflow as tf
from NETWORKS.vgg import build_vgg11, vgg11
from NETWORKS.lenet import build_lenet, lenet
from NETWORKS.alexnet import build_alexnet, alexnet
from NETWORKS.model_helper import evaluate_model, train_model, layers_to_transfer, load_model, transfer_model
from DATASETS.dataset_helper import get_mnist, get_svhn, get_cifar10
from report_helper import appendFile, checkDir

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


# -------------------------------------------------
# Datasets loading and preprocessing
print ("Data loading and preprocessing...")
# mX, mY, mtestX, mtestY = get_mnist(instances=50, rgb=True)
# sX, sY, stestX, stestY = mX, mY, mtestX, mtestY
# cX, cY, ctestX, ctestY = mX, mY, mtestX, mtestY
mX, mY, mtestX, mtestY = get_mnist(rgb=True)
sX, sY, stestX, stestY = get_svhn(crop=True)
cX, cY, ctestX, ctestY = get_cifar10(crop=True)


LEARN_RATE = 0.00001

# MODEL_PATH = './models/'
# CHECKPOINT_PATH = 'checkpoints/'
# TENSORBOARDDIR = 'tensorboard/'
# sudo mkdir /home/DLEx/
# sudo chmod 7777 /home/DLEx/
OUTPUT_PATH = '/home/DLEx'
MODEL_PATH = OUTPUT_PATH + '/models/'
CHECKPOINT_PATH = OUTPUT_PATH + '/checkpoints/'
TENSORBOARDDIR = OUTPUT_PATH + '/tensorboard/'
checkDir(MODEL_PATH)
checkDir(CHECKPOINT_PATH)
checkDir(TENSORBOARDDIR)


# # report file writer
def report_file(line, filename, net_name, ext):
    checkDir(OUTPUT_PATH + '/' + net_name + '/')
    file_path = OUTPUT_PATH + '/' + net_name + '/' + filename + '.' + ext + '.log'
    appendFile(file_path, [line])

# report_progress.max = number_of_experiments
# report_progress.file_name = get_time_str()
def report_progress(exec_name):
    report_progress.max -= 1
    status = str(report_progress.max) + " pending..."
    if report_progress.max == 0:
        status = 'ALL DONE!'
    print '%s (%s)'%(exec_name,status)
    file_path = OUTPUT_PATH + '/progress_' + report_progress.file_name + '.log'
    appendFile(file_path, [exec_name + ' | ' +status])


def get_time_str():
    return dt.now().strftime("%Hh%M-%d%m%Y")


def report_log_line(line):
    appendFile(REPORT_LOG_FILE_NAME, [line])


REPORT_LOG_FILE_NAME = 'run.log'


def report_log(RUN_ID, X, testX, t, EPOCHS, LEARN_RATE, acc):
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
    report_log_line(i)


# --------------------------------------------------------------
def pretrain_transfer(dataset, net_name, epochs):
    MODEL_NAME = 'model_' + dataset + '.tfl'
    RUN_ID = 'pretrain_' + dataset

    if net_name == "lenet":
        build_net = build_lenet
    elif net_name == "alexnet":
        build_net = build_alexnet
    elif net_name == "vgg11":
        build_net = build_vgg11

    if dataset == 'A':
        X, Y, testX, testY = mX, mY, mtestX, mtestY
    elif dataset == 'B':
        X, Y, testX, testY = sX, sY, stestX, stestY
    elif dataset == 'C':
        X, Y, testX, testY = cX, cY, ctestX, ctestY
    startCrono()
    # Training
    SAVE_PATH_FILE = MODEL_PATH + MODEL_NAME
    net = build_net(LEARN_RATE)
    model = train_model(net, X, Y,
                        epochs / 2,
                        SAVE_PATH_FILE,
                        RUN_ID,
                        CHECKPOINT_PATH,
                        TENSORBOARDDIR)
    t = getCrono()
    # evaluate model
    acc = evaluate_model(model, testX, testY)
    report_file(acc, RUN_ID, net_name, 'acc')
    report_file(t, RUN_ID, net_name, 'time')
    report_log(RUN_ID, X, testX, t, epochs / 2, LEARN_RATE, acc)
    tf.reset_default_graph()


def transfer(params_str, net_name, epochs):  # Dn_D
    sourceDataset = params_str[0]
    targetDataset = params_str[-1]
    transferParams = params_str[1:-1]

    if net_name == "lenet":
        net_num_layers = 5
    elif net_name == "alexnet":
        net_num_layers = 8
    elif net_name == "vgg11":
        net_num_layers = 11

    if (targetDataset == 'A'):
        X, Y, testX, testY = mX, mY, mtestX, mtestY
    elif (targetDataset == 'B'):
        X, Y, testX, testY = sX, sY, stestX, stestY
    elif (targetDataset == 'C'):
        X, Y, testX, testY = cX, cY, ctestX, ctestY

    trained_model_name = 'model_' + sourceDataset + '.tfl'
    transfer_exec(X, Y, testX, testY, trained_model_name, params_str, params_str + '.tfl',
                  layers_to_transfer(transferParams, net_num_layers), net_name, epochs)


def transfer_exec(X, Y, testX, testY, MODEL_NAME, RUN_ID, N_MODEL_NAME, transf_params_encoded, net_name, epochs):
    # transfer
    # load model
    startCrono()
    SAVE_PATH_FILE = MODEL_PATH + N_MODEL_NAME
    if net_name == "lenet":
        net_builder = lenet
    elif net_name == "alexnet":
        net_builder = alexnet
    elif net_name == "vgg11":
        net_builder = vgg11

    lmodel = load_model(net_builder,
                        MODEL_PATH,
                        MODEL_NAME,
                        LEARN_RATE,
                        CHECKPOINT_PATH,
                        TENSORBOARDDIR,
                        transf_params_encoded)
    lmodel = transfer_model(lmodel, X, Y,
                            epochs / 2,
                            RUN_ID,
                            SAVE_PATH_FILE)
    t = getCrono()
    # evaluate model
    acc = evaluate_model(lmodel, testX, testY)
    report_file(acc, RUN_ID, net_name, 'acc')
    report_file(t, RUN_ID, net_name, 'time')
    report_log(RUN_ID, X, testX, t, epochs / 2, LEARN_RATE, acc)
    tf.reset_default_graph()


def EXEC_TRANSFER(net_name, epochs):
    if net_name == "lenet":
        layer_index_list = ['1', '2', '3', '4']
    elif net_name == "alexnet":
        layer_index_list = ['1', '2', '3', '4', '5', '6', '7']
    elif net_name == "vgg11":
        layer_index_list = ['1', '2', '4', '6', '8', '10']
    #report progress init
    num_execs = len(layer_index_list)*3*3*2 + 3
    report_progress.file_name = get_time_str()
    report_progress.max = num_execs
    #
    for D in ['A', 'B', 'C']:
        pretrain_transfer(D, net_name, epochs)
        report_progress('pre_train_'+D)
    for S in ['A', 'B', 'C']:
        for T in ['A', 'B', 'C']:
            for layer in layer_index_list:
                for mode in ['+', '-']:
                    transfer(S + layer + mode + T, net_name, epochs)
                    report_progress(S + layer + mode + T)


def run_exp(model, epochs, exps):
    print "Experiments with:"
    print "Model", model
    print "Epochs = ", epochs
    print "Experiments = ", exps
    for i in range(int(exps)):
        EXEC_TRANSFER(model, int(epochs))
    sout('END!')