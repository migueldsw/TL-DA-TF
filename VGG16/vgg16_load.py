# -*- coding: utf-8 -*-
'''
Retraining (Finetuning) Example with vgg.tflearn. Using weights from VGG-11 model to retrain
network for a new task.All weights are restored except
last layer (softmax) that will be retrained to match the new task (finetuning).
'''
import tflearn
import os


def vgg16(input, num_class):

    x = tflearn.conv_2d(input, 8, 3, activation='relu', scope='conv1_1')
    x = tflearn.conv_2d(input, 8, 3, activation='relu', scope='conv1_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 16, 3, activation='relu', scope='conv2_1')
    x = tflearn.conv_2d(x, 16, 3, activation='relu', scope='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.conv_2d(x, 32, 3, activation='relu', scope='conv3_1')
    x = tflearn.conv_2d(x, 32, 3, activation='relu', scope='conv3_2')
    x = tflearn.conv_2d(x, 32, 3, activation='relu', scope='conv3_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv4_1')
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv4_2')
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv4_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv5_1')
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv5_2')
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = tflearn.fully_connected(x, 512, activation='relu', scope='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    x = tflearn.fully_connected(x, 512, activation='relu', scope='fc7')
    x = tflearn.dropout(x, 0.5, name='dropout2')

    x = tflearn.fully_connected(x, num_class, activation='softmax', scope='fc8', restore=False)

    return x


model_path = "./models/"

from dataset_helper import get_mnist
X, Y, testX, testY = get_mnist(rgb=True)


NUM_CLASSES = 10
LEARNING_RATE = 0.0001
EPOCH = 1

# VGG Network
inp = tflearn.input_data(shape=[None, 28, 28, 3], name='input')

softmax = vgg16(inp, NUM_CLASSES)
regression = tflearn.regression(softmax, optimizer='rmsprop',
                                loss='categorical_crossentropy',
                                learning_rate=LEARNING_RATE, restore=False)

model = tflearn.DNN(regression, checkpoint_path='vgg-finetuning',
                    max_checkpoints=3, tensorboard_verbose=2,
                    tensorboard_dir="./logs")

model_file = os.path.join(model_path, "vgg16-model1.tfl")
model.load(model_file, weights_only=True)

# Start finetuning
model.fit(X, Y, n_epoch=EPOCH, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_epoch=False,
          snapshot_step=200, run_id='vgg-finetuning')

# Save the model
model.save('./models/vgg16-retrained-1')
print ('Model SAVED!')

# Evaluate accuracy.
accuracy_score = model.evaluate(testX,testY,batch_size=32)
print('Accuracy: %s' %accuracy_score)