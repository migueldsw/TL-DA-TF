#!/bin/sh
mkdir BKP
mkdir BKP/$1
cp -R ./checkpoints/ ./BKP/$1/checkpoints
cp -R ./models/ ./BKP/$1/models
cp -R ./outexec.log ./BKP/$1/outexec.log
cp -R ./run.log ./BKP/$1/run.log
cp -R ./tensorboard/ ./BKP/$1/tensorboard

rm -r ./checkpoints/
rm -r ./models/
rm -r ./outexec.log
rm -r ./run.log
rm -r ./tensorboard/