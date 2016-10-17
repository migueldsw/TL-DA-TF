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
pip install -U scikit-learn

#install tensorflow
#tensorflow v0.11
#GPU - CUDA: 
#export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl
#CPU:
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL
