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
import time
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
        z = Theta.dot(features).reshape((Theta.shape[0],1))
        a = Ulti.sigmoid(z)
        zs_as_per_layer.append((z,a))
        if (i == len(Thetas) - 1):
            return np.array(zs_as_per_layer)
        a = np.insert(a, 0, 1) # Add the bias unit
        features = a

# print(computeCost(Ulti.flattenParams(myThetas, input_layer_size, hidden_layer_size, output_layer_size), Ulti.flattenX(X,n_training_sample,input_layer_size), y))

def genRandomThetas():
    epsilon_init = 0.12 
    theta1_shape = (hidden_layer_size, input_layer_size + 1)
    theta2_shape = (output_layer_size, hidden_layer_size + 1)
    return [ np.random.rand(*theta1_shape)*2*epsilon_init - epsilon_init, np.random.rand(*theta2_shape)*2*epsilon_init - epsilon_init]

def backPropagate(myThetas_flattened, myX_flattened, myY, myLamda = .0):
    # First unroll the params
    _myThetas = Ulti.reshapeParams(myThetas_flattened, input_layer_size, hidden_layer_size, output_layer_size)
    # Unroll X
    _myX = Ulti.reshapeX(myX_flattened, n_training_sample, input_layer_size)

    m = n_training_sample

    # The Delta matrices should include the bias unit
    # The Delta matrices have the same shape as the theta matrices
    Delta1 = np.zeros((hidden_layer_size, input_layer_size + 1))
    Delta2 = np.zeros((output_layer_size, hidden_layer_size + 1))

    for irow in range(m):
        myRow = _myX[irow]

        a1 = myRow.reshape((input_layer_size + 1, 1))

        # propagateForward returns (zs, activations) for each layer excluding the input layer
        temp = propagateForward(myRow, _myThetas)

        z2 = temp[0][0]
        a2 = temp[0][1]
        z3 = temp[1][0]
        a3 = temp[1][1]

        tmpy = np.zeros((10,1))
        k = myY[irow] - 1
        tmpy[k] = 1

        delta3 = a3 - tmpy
        delta2 = _myThetas[1].T[1:,:].dot(delta3)*(Ulti.sigmoidGradient(z2)) #remove the 0th element in hidden layer

        a2 = np.insert(a2, 0, 1, axis=0)

        Delta1 += delta2.dot(a1.T) # (25,1)*(1*401) = (25*401)
        Delta2 += delta3.dot(a2.T) # (10,1)*(1*26) = (10,26)

    D1, D2 = Delta1/float(m), Delta2/float(m)

    # Gradient Regularization
    D1[:,1:] = D1[:,1:] + (float(myLamda)/m)*_myThetas[0][:,1:]
    D2[:,1:] = D2[:,1:] + (float(myLamda)/m)*_myThetas[1][:,1:]

    return Ulti.flattenParams([D1, D2], input_layer_size, hidden_layer_size, output_layer_size).flatten()

flattendD1D2 = backPropagate(Ulti.flattenParams(myThetas, input_layer_size, hidden_layer_size, output_layer_size), Ulti.flattenX(X,n_training_sample,input_layer_size), y, myLamda=.0)

D1, D2 = Ulti.reshapeParams(flattendD1D2, input_layer_size, hidden_layer_size, output_layer_size)

def checkingGradient(Thetas, myDs, myX, myY, myLamda = .0):
    # this value is picked in the Document
    myEsp = 10e-4
    # flatten input
    flattened = Ulti.flattenParams(Thetas, input_layer_size, hidden_layer_size, output_layer_size)
    flattenedDs = Ulti.flattenParams(myDs, input_layer_size, hidden_layer_size, output_layer_size)
    myX_flattened = Ulti.flattenX(myX, n_training_sample, input_layer_size)
    n_elems = len(flattened)
    # pick random 10 elements
    for i in range(10):
        x = int(np.random.rand()*n_elems)
        epsvec = np.zeros((n_elems, 1))
        epsvec[x] = myEsp

        cost_high = computeCost(flattened + epsvec, myX_flattened, myY, myLamda)
        cost_low = computeCost(flattened - epsvec, myX_flattened, myY, myLamda)

        myGrad = (cost_high - cost_low)/(2*myEsp)
        print("Element: %d. Numerical Gradient = %f. BackProp Gradient = %f"%(x, myGrad, flattenedDs[x]))

checkingGradient(myThetas, [D1, D2], X, y)

# Learning params using fmin_cg

def trainNN(myLamda = 5.0):
    """
    Function that generates random initial theta matrices, optimize them 
    and return a list of reshaped theta matrices
    """
    started_time = time.time()
    random_thetas = Ulti.flattenParams(genRandomThetas(), input_layer_size, hidden_layer_size, output_layer_size)

    result = scipy.optimize.fmin_cg(computeCost, x0 = random_thetas, fprime=backPropagate, 
                args=(Ulti.flattenX(X, n_training_sample, input_layer_size), y, myLamda), maxiter=50, disp=True, full_output=True)

    print("Done training in: %i seconds"%(time.time() - started_time))
    return Ulti.reshapeParams(result[0], input_layer_size, hidden_layer_size, output_layer_size)

learnt_theta = trainNN()

def predictNN(row, Thetas):
    """
    Function that takes a row of features, propagates them through the
    NN, and returns the predicted integer that was hand written
    """
    classes = [i for i in range(1,11)]

    output = propagateForward(row, Thetas)

    return classes[np.argmax(output[-1][1])]

def computeAccuracy(myX, myThetas, myY):
    """
    Function that loops over all of the rows in X (all of the handwritten images)
    and predicts what digit is written given the thetas. Check if it's correct, and
    compute an efficiency.
    """
    n_correct, n_total = 0, myX.shape[0]
    for irow in range(n_total):
        if (predictNN(myX[irow], myThetas) == int(myY[irow])):
            n_correct += 1

    print("Training set accuracy: %f"%(float(n_correct)/n_total))

computeAccuracy(X, myThetas, y)
        
def displayHiddenUnit(myTheta):
    """
    """
    # Remote bias unit
    myTheta = myTheta[:,1:]

    assert myTheta.shape == (25,400)

    width, height = 20, 20
    nrows, ncols = 5, 5

    big_picture = np.zeros((width*ncols,height*nrows))

    irow, icol = 0, 0
    for row in myTheta:
        if (icol == ncols):
            irow += 1
            icol = 0
        # Add back bias unit
        iimg = Ulti.getDatumImg(np.insert(row,0,1))
        big_picture[irow*height:irow*height+iimg.shape[0],icol*width:icol*width+iimg.shape[1]] = iimg
        icol += 1
    fig = plt.figure(figsize=(6,6))
    img = scipy.misc.toimage(big_picture)
    plt.imshow(img,cmap = 'gray')
    plt.show()

displayHiddenUnit(learnt_theta[0])