from __future__ import print_function
import numpy as np
from sklearn.datasets import fetch_mldata
import tensorflow.examples.tutorials.mnist.input_data as input_data

# dataDir = './data'
# mnist = fetch_mldata('MNIST Original', data_home=dataDir)
mnist = input_data.read_data_sets('MNIST')
print('Shape of mnist data', mnist.data.shape)
