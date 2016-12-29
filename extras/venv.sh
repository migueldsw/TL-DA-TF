#!/bin/sh

sudo pip install virtualenvwrapper
sudo export WORKON_HOME=~/Envs
sudo mkdir -p $WORKON_HOME
sudo source /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv env1
workon env1