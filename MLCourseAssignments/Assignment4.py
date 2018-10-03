from __future__ import print_function

from PIL import Image
import sys
import random
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import io as io
from scipy import optimize
import itertools
import Ulti

filePathTheta = './DataAssignment3/ex3weights.mat'
filePath = './DataAssignment3/ex3data1.mat'

# Check if in debug mode
gettrace = getattr(sys, 'gettrace', None)

if gettrace():
    print('In Debug Mode!')
    filePathTheta = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment3\ex3weights.mat'
    filePath = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment3\ex3data1.mat'

mat_t = io.loadmat(filePathTheta)
mat = io.loadmat(filePath)

Theta1, Theta2 = mat_t['Theta1'], mat_t['Theta2']
X, y = mat['X'], mat['y']

myThetas = [Theta1, Theta2]

X = np.insert(X, 0, 1, axis=1)

print("Theta1 has shape:",Theta1.shape)
print("Theta2 has shape:",Theta2.shape)

# Ulti.displayData(X)

# Model Representation

# These are not including bias units
input_layer_size = 400
hidden_layer_size = 25
output_layer_size = 10

# Some utility functions. There are lot of flattening and
# reshaping of theta matrices, the input X matrix, etc...
# Nicely shaped matrices make the linear algebra easier when developing

def flattenParams(thetas_list):
    """
    Hand this function a list of theta matrices and it will flatten it 
    into one long (n, 1) shaped numpy array
    """
    # flatten an numpy array
    # ex1: np.array([[1], [3], [5,6]]) => [list(1), list(3), list(5,6)]
    # ex2: np.array([[1], [3], [5]]) => [1, 3, 5]
    flattened_list = [ myTheta.flatten() for myTheta in thetas_list ]
    # combine all sub array in an array into a big array
    combined = list(itertools.chain.from_iterable(flattened_list))

    assert len(combined) == (input_layer_size + 1)*hidden_layer_size + (hidden_layer_size + 1)*output_layer_size

    return np.array(combined).reshape((len(combined),1))


