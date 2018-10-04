from __future__ import print_function

import random
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

def reshapeParams(flattened_array):
	theta1 = fl
