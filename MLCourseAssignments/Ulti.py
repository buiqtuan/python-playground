from __future__ import print_function

import random
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools

def getDatumImg(row):
	"""
    Function that is handed a single np array with shape 1x400,
    reshape it into 20x20 numpy array then transpose it
    """
	width,height = 20, 20
	square = row[1:].reshape(width,height)
	return square.T

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def h(myT, myX):
    return sigmoid(np.dot(myX, myT))

def displayData(X, indices_to_display = None):
	"""
    Function that picks 100 random rows from X, creates a 20x20 image from each,
    then stitches them together into a 10x10 grid of images, and shows it.
    """
	width, height = 20, 20
	nrows, ncols = 10, 10
	if not indices_to_display:
		indices_to_display = random.sample(range(X.shape[0]), nrows*ncols)

	big_picture = np.zeros((height*nrows, width*ncols))

	irow, icol = 0, 0
	# Fill the image array to big array
	for idx in indices_to_display:
		if (icol == ncols):
			irow += 1
			icol  = 0
		
		iimg = getDatumImg(X[idx])
		# Only with numpy array
		big_picture[irow*height:irow*height+iimg.shape[0],icol*width:icol*width+iimg.shape[1]] = iimg
		icol += 1

	fig = plt.figure(figsize=(10,10))
	img = scipy.misc.toimage(big_picture)
	plt.imshow(img,cmap = 'gray')
	plt.show()

# Some utility functions. There are lot of flattening and
# reshaping of theta matrices, the input X matrix, etc...
# Nicely shaped matrices make the linear algebra easier when developing
def flattenParams(thetas_list, input_layer_size, hidden_layer_size, output_layer_size):
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

def reshapeParams(flattened_array, input_layer_size, hidden_layer_size, output_layer_size):
	"""
	This fucntion reshape a flattened array into the original theta 1 and 2
	"""
	theta1 = flattened_array[:(input_layer_size + 1)*hidden_layer_size].reshape((hidden_layer_size, input_layer_size + 1))

	theta2 = flattened_array[(input_layer_size + 1)*hidden_layer_size:].reshape((output_layer_size, hidden_layer_size + 1))

	return [theta1, theta2]

def flattenX(myX, n_training_samples, input_layer_size):
	"""
	(m input, each input is (1,n) vector) => a ((m*(n+1)),1) vector
	"""
	return np.array(myX.flatten()).reshape((n_training_samples*(input_layer_size + 1), 1))

def reshapeX(flattenedX, n_training_samples, input_layer_size):
	"""
	a flatten array (1, m*(n+1)) => (m, n+ 1) array
	"""
	return np.array(flattenedX).reshape(n_training_samples, input_layer_size + 1)

def sigmoidGradient(x):
	g = sigmoid(x)
	return np.array(g*(1 - g))