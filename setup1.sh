#!/bin/sh
echo "SETUP 1"
echo "Script executed from: ${PWD}"
cd ~
echo "Now at: ${PWD}"
export LC_ALL=C

sudo apt-get --assume-yes install virtualenv
sudo apt-get --assume-yes install python-pip

#virtualenv
virtualenv --system-site-packages ~/tensorflow
source ~/tensorflow/bin/activate

#python data sci. libs
sudo apt-get --assume-yes install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
sudo pip install -U scikit-learn
sudo pip install tflearn

#install tensorflow
#obs.: # GPU Requires CUDA toolkit 7.5 and CuDNN v5.
#tensorflow v0.11
if [$1 = 0.11] && [$2 = gpu]
	then
	#GPU - CUDA: 
	export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl
	echo "Will install TensorFlow v0.11 GPU"
elif [$1 = 0.11] && [$2 = cpu]
	then
	#CPU:
	export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl
	echo "Will install TensorFlow v0.11 CPU"
#tensorflow v0.10
elif [$1 = 0.10] && [$2 = cpu]
	then
	#CPU:
	export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
	echo "Will install TensorFlow v0.10 CPU"
elif [$1 = 0.10] && [$2 = gpu]
	then
	#GPU:
	export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
	echo "Will install TensorFlow v0.10 GPU"
else
	echo "Cant find tensorflow version: $1 $2"
	export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl
	echo "Will install TensorFlow v0.11 CPU (DEFAULT)"
fi
sudo pip install --upgrade $TF_BINARY_URL
