#!/bin/sh
mkdir BKP
mkdir BKP/$1
cp -R ./checkpoints/ ./BKP/$1/checkpoints
cp -R ./models/ ./BKP/$1/models
cp -R ./outexec.log ./BKP/$1/outexec.log
cp -R ./run.log ./BKP/$1/run.log
cp -R ./tensorboard/ ./BKP/$1/tensorboard
cp -R ./tensorboard/ ./BKP/$1/lenet
cp -R ./tensorboard/ ./BKP/$1/alexnet
cp -R ./tensorboard/ ./BKP/$1/vgg11
cp -R ./tensorboard/ ./BKP/$1/vgg16

rm -r ./checkpoints/
rm -r ./models/
rm -r ./outexec.log
rm -r ./run.log
rm -r ./tensorboard/
rm -r ./lenet/
rm -r ./alexnet/
rm -r ./vgg11/
rm -r ./vgg16/