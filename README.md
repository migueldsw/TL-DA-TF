# TL-DA-TF
Transfer Learning in Deep Architectures using TFLearn -> (Google)TensorFlow 

## Networks
Adjusted networks for 28x28x3 image inputs. Code for pre training and training (fine tuning).
### VGG-11 VGG-16
- [VGG-11/16](https://github.com/migueldsw/TL-DA-TF/blob/master/NETWORKS/vgg.py): 16 and 11 layers. 
### AlexNet
- [AlexNet](https://github.com/migueldsw/TL-DA-TF/blob/master/NETWORKS/alexnet.py): 8 layers.
### LeNet-5
- [LeNet-5](https://github.com/migueldsw/TL-DA-TF/blob/master/NETWORKS/lenet.py): 5 layers. 

## Datasets
Source and target datasets explored for transfer learning combinations.
### MNIST
RGB channels included (28x28x3)
### SVHN
Images cropped to 28x28x3
### CIFAR-10
Images cropped to 28x28x3
