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
n_training_sample = X.shape[0]

def computeCost(myThetas_flattened, myX_flattened, myY, myLamda = .0):
    """
    This function takes in:
        1) a flattened vector of theta parameters (each theta would go from one
           NN layer to the next), the thetas include the bias unit.
        2) the flattened training set matrix X, which contains the bias unit first column
        3) the label vector y, which has one column
    It loops over training points (recommended by the professor, as the linear
    algebra version is "quite complicated") and:
        1) constructs a new "y" vector, with 10 rows and 1 column, 
            with one non-zero entry corresponding to that iteration
        2) computes the cost given that y- vector and that training point
        3) accumulates all of the costs
        4) computes a regularization term (after the loop over training points)
    """
    # First unroll the params
    _myThetas = Ulti.reshapeParams(myThetas_flattened, input_layer_size, hidden_layer_size, output_layer_size)
    # Unroll X
    _myX = Ulti.reshapeX(myX_flattened, n_training_sample, input_layer_size)
    # Accumulate the total cost
    total_cost = 0

    m = n_training_sample

    # Loop over the training point
    for irow in range(m):
        myRow = _myX[irow]

        # First compute the hypothesis (this is a (10,1) vector
        # of the hypothesis for each possible y-value)
        # propagateForward returns (zs, activations) for each layer
        # so propagateforward[-1][1] means "activation for -1st (last) layer"
        myHs = propagateForward(myRow, _myThetas)[-1][1]

        # Construct a 10x1 "y" vector with all zeros and only one "1" entry
        # note here if the hand-written digit is "0", then that corresponds
        # to a y- vector with 1 in the 10th spot (different from what the
        # homework suggests)
        tmpy = np.zeros((10,1))
        k = myY[irow] - 1
        tmpy[k] = 1

        # Compute the cost for this point and y-vector
        myCost = -tmpy.T.dot(np.log(myHs)) - (1 - tmpy.T).dot(np.log(1 - myHs))

        # total cost
        total_cost += myCost

    total_cost = float(total_cost/m)

    # compute reg term
    total_reg = .0

    for _myTheta in _myThetas:
        total_reg += np.sum(_myTheta*_myTheta) # element-wise multiplication

    total_reg *= float(myLamda)/(2*m)

    return total_cost + total_reg

def propagateForward(row, Thetas):
    """
    Function that given a list of Thetas (NOT flattened), propagates the
    row of features forwards, assuming the features ALREADY
    include the bias unit in the input layer, and the 
    Thetas also include the bias unit

    The output is a vector with element [0] for the hidden layer,
    and element [1] for the output layer
        -- Each element is a tuple of (zs, as)
        -- where "zs" and "as" have shape (# of units in that layer, 1)
    
    ***The 'activations' are the same as "h", but this works for many layers
    (hence a vector of thetas, not just one theta)
    Also, "h" is vectorized to do all rows at once...
    this function takes in one row at a time***
    """
    features = row
    zs_as_per_layer = []
    for i in range(len(Thetas)):
        Theta = Thetas[i]
        # Theta is (25, 401), features are (401,1)
        # So z comes out to be (25,1)
        # This is one z value for each unit in the hidden layer
        # Not counting the bias unit
        z = Theta.dot(features)
        a = Ulti.sigmoid(z)
        zs_as_per_layer.append((z,a))
        if (i == len(Thetas) - 1):
            return np.array(zs_as_per_layer)
        a = np.insert(a, 0, 1) # Add the bias unit
        features = a

print(computeCost(Ulti.flattenParams(myThetas, input_layer_size, hidden_layer_size, output_layer_size), Ulti.flattenX(X,n_training_sample,input_layer_size), y))