#Dataset Helper
#

import tflearn
import numpy as np

# import tflearn.datasets.mnist as mnist
# X, Y, testX, testY = mnist.load_data(one_hot=True)

#reshape a from 784x1 to 28x28
#a = a.reshape(28,28)

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

def get_small_mnist(instances):
	import tflearn.datasets.mnist as mnist
	X, Y, testX, testY = mnist.load_data(one_hot=True)
	x, y = small_set(X,Y,instances,10)
	tx, ty = small_set(testX,testY,instances/10,10)
	if (len(tx)<1):
		tx, ty = testX[-3:], testY[-3:]
	return x, y, tx, ty

def get_small_svhn(instances):
	import tflearn.datasets.svhn as svhn
	X, Y, testX, testY = svhn.load_data(one_hot=True)
	x, y = small_set(X,Y,instances,10)
	tx, ty = small_set(testX,testY,instances/10,10)
	if (len(tx)<1):
		tx, ty = testX[-3:], testY[-3:]
	return x, y, tx, ty

def get_small_oxf17(instances):
	import tflearn.datasets.oxflower17 as oxflower17
	fX, fY = oxflower17.load_data(one_hot=True)
	val = 136
	trainX, trainY = fX[:val], fY[:val]
	testX, testY = fX[val:], fY[val:]
	x, y = small_set(trainX, trainY,instances,17)
	tx, ty = small_set(testX,testY,instances/17,17)
	if (len(tx)<1):
		tx, ty = testX[-3:], testY[-3:]
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
