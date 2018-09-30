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

X = np.insert(X, 0, 1, axis=1)

print("Theta1 has shape:",Theta1.shape)
print("Theta2 has shape:",Theta2.shape)

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

def propagateForward(row, Thetas):
    """
    Function that given a list of Thetas, propagates the
    Row of features forwards, assuming the features already
    include the bias unit in the input layer, and the 
    Thetas need the bias unit added to features between each layer
    """
    Theta = Thetas[0] # shape: (25, 401)
    z = Theta.dot(row) # shape: (25, 1)
    a = sigmoid(z) # shape: (25, 1)
    a = np.insert(a,0,1) # shape: (26, 1)

    Theta = Thetas[1] # shape: (10, 26)
    _z = Theta.dot(a) # shape: (10, 1)
    _a = sigmoid(_z) # shape: (10, 1)

    return _a

# Same as propagateForward but is written in another way
def _propagateForward(row,Thetas):
    features = row

    for i in range(len(Thetas)):
        Theta = Thetas[i] # shape: (25, 401)
        z = Theta.dot(features) # shape: (25, 1)
        a = sigmoid(z) # shape: (25, 1)
        if (i == (len(Thetas) - 1)):
            return a
        a = np.insert(a, 0, 1)# Add the bias unit
        features = a

def predictNN(row, Thetas):
    """
    Function that takes a row of features, propagates them through the
    NN, and returns the predicted integer that was hand written
    """
    classes = []
    for i in range(1,11):
        classes.append(i)
    output = propagateForward(row,Thetas)
    return classes[np.argmax(np.array(output))]

myThetas = [ Theta1, Theta2 ]
n_correct, n_total = 0., 0.
incorrect_indices = []
#Loop over all of the rows in X (all of the handwritten images)
#and predict what digit is written. Check if it's correct, and
#compute an efficiency.

for irow in range(X.shape[0]):
	n_total +=1
	if (predictNN(X[irow], myThetas) == int(y[irow])):
		n_correct +=1
	else:
		incorrect_indices.append(irow)
print("Training set accuracy: %0.1f%%"%(100*(n_correct/n_total)))

print('Show some random incorrect predictions: ')

for i in range(5):
    i = random.choice(incorrect_indices) # pick a random element in a equence
    fig = plt.figure(figsize=(3,3))
    img = scipy.misc.toimage(getDatumImg(X[i]))
    plt.imshow(img, cmap= 'gray')
    predicted_val = predictNN(X[i], myThetas)
    predicted_val = 0 if predicted_val == 10 else predicted_val
    fig.suptitle('Predicted: %d'%predicted_val, fontsize=14, fontweight='bold')
    plt.show()