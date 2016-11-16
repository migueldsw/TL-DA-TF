#Dataset Helper
#

import tflearn
import numpy as np


def label_of(one_hot_vector):
	return one_hot_vector.argmax()

def small_set(X,Y,instances,classes):
	instances_of_class = instances/classes
	counter = np.zeros(classes)
	#add one instance of one class to counter
	index_list = []
	c = 0
	while (np.sum(counter) < instances and c < len(X)):
		label = label_of(Y[c])
		if(counter[label] < instances_of_class):
			counter[label]+=1
			index_list.append(c)
		c+=1
	#prepare output
	nX = []
	nY = []
	for i in index_list:
		nX.append(X[i])
		nY.append(Y[i])
	return np.array(nX), np.array(nY)

def get_mnist(instances = None, rgb = False, see = False):
	import tflearn.datasets.mnist as mnist
	X, Y, testX, testY = mnist.load_data(one_hot=True)
	X = X.reshape(len(X),28,28,1)
	testX = testX.reshape(len(testX),28,28,1)
	x, y, tx, ty = X, Y, testX, testY
	if (instances):
		x, y = small_set(X,Y,instances,10)
		tx, ty = small_set(testX,testY,instances/10,10)
		if (len(tx)<1):
			tx, ty = testX[-3:], testY[-3:]
	if (rgb): 
		#include the RGB layers in sets
		[x, tx] = map(include_RBG_in_set,[x, tx])
	if (see):
		see_image(x[0])
	return x, y, tx, ty

def get_svhn(instances = None, crop = False, see = False):
	import tflearn.datasets.svhn as svhn
	X, Y, testX, testY = svhn.load_data(one_hot=True)
	x, y, tx, ty = X, Y, testX, testY
	if (instances):
		x, y = small_set(X,Y,instances,10)
		tx, ty = small_set(testX,testY,instances/10,10)
		if (len(tx)<1):
			tx, ty = testX[-3:], testY[-3:]
	if (crop): 
		#crop images in output sets
		[x, tx] = map(crop_images_in_set,[x, tx])
	if (see):
		see_image(x[0])
	return x, y, tx, ty

def get_svhn_extra(crop = False, see = False):
	import tflearn.datasets.svhn as svhn
	X, Y, testX, testY = svhn.load_data(one_hot=True)
	X, Y= svhn.load_extra_data(one_hot=True)
	x, y, tx, ty = X, Y, testX, testY
	if (crop): 
		[x, tx] = map(crop_images_in_set,[x, tx])
	if (see):
		see_image(x[0])
	return x, y, tx, ty

def get_cifar10(instances = None, crop = False, see = False):
	import tflearn.datasets.cifar10 as cifar10
	(X, Y), (testX, testY) = cifar10.load_data(dirname="cifar10", one_hot=True)
	x, y, tx, ty = X, Y, testX, testY
	if (instances):
		x, y = small_set(X,Y,instances,10)
		tx, ty = small_set(testX,testY,instances/10,10)
		if (len(tx)<1):
			tx, ty = testX[-3:], testY[-3:]
	if (crop): 
		#crop images in output sets
		[x, tx] = map(crop_images_in_set,[x, tx])
	if (see):
		see_image(x[0])
	return x, y, tx, ty

def see_image(data):
	from matplotlib import pyplot as plt
	plt.imshow(data, interpolation='nearest')
	plt.show()

def include_RGB_layers(img):
	#reshapes the input img from 28x28x1 to 28x28x3
	out = np.zeros(28*28*3).reshape(28,28,3)
	for i in range(28):
		for j in range(28):
			out[i,j] = np.ones(3)*img[i,j,0]
	return out

def include_RBG_in_set(X):
	#reshapes de input set X from Nx28x28x1 to Nx28x28x3
	out = np.zeros(np.prod(X.shape[:-1]+(3,))).reshape(X.shape[:-1]+(3,))
	for i in range(len(X)):
		out[i] = include_RGB_layers(X[i])
	return out

def crop_img(data):
	#data shape: (32x32x3) crop to (28x28x3)
	ndata = np.zeros(28*28*3).reshape(28,28,3)
	for rgb in range(3):
		for i in range(28):
			for j in range(28):
				ndata[i,j,rgb] = data[i+2,j+2,rgb]
	return ndata

def crop_images_in_set(X):
	#crops images in set X from shape (32x32x3) to new shape(28x28x3)
	nX = np.zeros(len(X)*28*28*3).reshape(len(X),28,28,3)
	for i in range(len(X)):
		nX[i] = crop_img(X[i])
	return nX