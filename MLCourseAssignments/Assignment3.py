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

def costFucntion(myTheta, myX, myY, myLamda = 0.):
	m = myTheta.shape[0] # 5000
	myH = h(myTheta, myX) # shape: (5000,1)
	term1 = np.dot(- myY.T, np.log(myH)) # shape: (5000, 5000)
	term2 = np.dot((1.0 - myY), np.log(1.0 - myH)) # shape: (5000, 5000)

	left_hand = term1 - term2 # shape: (5000, 5000)
	# Add Regularized factor
	right_hand = myTheta.T.dot(myTheta) * (myLamda/(2*m)) # shape: (1, 1)

	return left_hand + right_hand # shape: (5000, 5000)

#An alternative to OCTAVE's 'fmincg' we'll use some scipy.optimize function, "fmin_cg"
#This is more efficient with large number of parameters.
#fmin_cg needs the gradient handed do it
def costGradient(myTheta, myX, myY, myLamda = 0.):
	m = myX.shape[0]

	# Transpose Y because the input Y is given in shape (1,5000), no theoretical reason here, just to fit with input data
	beta = h(myTheta, myX) - myY.T # Share: (5000, 1)

	regTerm = myTheta[1:]*(myLamda/m) # shape: (400, 1)

	grad = (1./m)*np.dot(myX.T, beta) # share: (401, 1)

	grad[1:] = grad[1:] + regTerm

	return grad # shape: (401,1)

# increase
def optimizeTheta(myTheta, myX, myY, myLamda = 0.):
	result = optimize.fmin_cg(costFucntion, fprime=costGradient, 
			x0=myTheta,args=(myX, myY, myLamda), maxiter=80, disp=False, full_output=True)

	return result[0], result[1]


def buildTheta():
	"""
    Function that determines an optimized theta for each class
    and returns a Theta function where each row corresponds
    to the learned logistic regression params for one class
    """
	myLamda = 0.
	initial_theta = np.zeros((X.shape[1],1)).reshape(-1)
	Theta = np.zeros((10, X.shape[1]))
	for i in range(10):
		iclass = i if i else 10 # class 10 corresponds to zero
		print("Optimizing for handwritten number %d..."%i)
		logic_Y = np.array([1 if x == iclass else 0 for x in y]) # 1 for the class is currently in training, 0 for 9 other classes
		itheta, imincost = optimizeTheta(initial_theta, X, logic_Y, myLamda)
		Theta[i,:] = itheta
	print("Done!")
	return Theta

Theta = buildTheta()

def predictOneVsAll(myTheta, myRow):
	classes = [10]
	for i in range(1,10):
		classes.append(i)

	hypots = [0]*len(classes)
	#Compute a hypothesis for each possible outcome
    #Choose the maximum hypothesis to find result
	for i in range(len(classes)):
		hypots[i] = h(myTheta[i], myRow)

	# argmax return the index or max element in the numpy array
	return classes[np.argmax(np.array(hypots))]

n_correct, n_total = 0., 0.
incorrect_indices = []

for irow in range(X.shape[0]):
	n_total +=1
	if (predictOneVsAll(Theta, X[irow]) == y[irow]):
		n_correct +=1
	else:
		incorrect_indices.append(irow)
print("Training set accuracy: %0.1f%%"%(100*(n_correct/n_total)))

# display image that have been classified wrong
# displayData(incorrect_indices[:100])
# displayData(incorrect_indices[100:200])
# displayData(incorrect_indices[200:300])