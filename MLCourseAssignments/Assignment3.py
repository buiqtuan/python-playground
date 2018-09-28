from __future__ import print_function

from PIL import Image
import sys
import random
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import io as io

filePath = './DataAssignment3/ex3data1.mat'

# Check if in debug mode
gettrace = getattr(sys, 'gettrace', None)

if gettrace():
	print('In Debug Mode!')
	filePath = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment3\ex3data1.mat'

mat = io.loadmat(filePath)

X, y = mat['X'], mat['y']

X = np.insert(X,0,1, axis=1)
print("'y' shape: %s. Unique elements in y: %s"%(mat['y'].shape,np.unique(mat['y'])))
print("'X' shape: %s. X[0] shape: %s"%(X.shape,X[0].shape))

def getDatumImg(row):
	"""
    Function that is handed a single np array with shape 1x400,
    reshape it into 20x20 numpy array then transpose it
    """
	width,height = 20, 20
	square = row[1:].reshape(width,height)
	return square.T

def displayData(indices_to_display = None):
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

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def h(myT, myX):
    return sigmoid(np.dot(myX, myT)) 