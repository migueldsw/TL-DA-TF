"""
LeNet-5

Ref.: Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.
Link: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
"""

from __future__ import division, print_function, absolute_import
import sys
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

keep_prob = 0.5

#Build LeNet-5
def build_lenet5(learning_rate, n_classes = 10):
    network = input_data(shape=[None,28,28,3], name='input')

    network = conv_2d(network,)