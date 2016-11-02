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
	return x, y, tx, ty

