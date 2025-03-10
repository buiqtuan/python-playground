from __future__ import print_function

import sys
import random
import numpy as np
import scipy
from scipy import io as io
from scipy import optimize
import matplotlib.pyplot as plt
import time

filePath = './DataAssignment5/ex5data1.mat'

# Check if in debug mode
gettrace = getattr(sys, 'gettrace', None)

if gettrace():
    print('In Debug Mode!')
    filePath = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment5\ex5data1.mat'

mat = io.loadmat(filePath)

# Training set
X, Y = mat['X'], mat['y']
# Cross validation set
Xval, Yval = mat['Xval'], mat['yval']
# Test set
Xtest, Ytest = mat['Xtest'], mat['ytest']
# Insert 1 as bias unit to each set
X = np.insert(X, 0, 1, axis=1)
Xval = np.insert(Xval, 0, 1, axis=1)
Xtest = np.insert(Xtest, 0, 1, axis=1)

# print(X.shape)
# print(Xval.shape)
# print(Xtest.shape)

def plotData():
    plt.figure(figsize=(8,5))
    plt.ylabel('Water flowing out of the dam (y)')
    plt.xlabel('Change in water level (x)')
    plt.plot(X[:,1], Y, 'rx')
    plt.grid(True)

# plotData()

def h(theta, X):
    return np.dot(X, theta)

def computeCost(myTheta, myX, myY, myLamda = .0):
    """
    theta_start is an n- dimensional vector of initial theta guess
    X is matrix with n- columns and m- rows
    y is a matrix with m- rows and 1 column
    """
    m = myX.shape[0]
    myH = h(myTheta, myX).reshape((m,1))
    myCost = float(1./(2*m)) * np.dot((myH - myY).T, (myH - myY))
    regTerm = (float(myLamda)/(2*m)) * float(myTheta[1:].T.dot(myTheta[1:]))

    return myCost + regTerm

myTheta = np.array([[1.],[1.]])

# print(computeCost(myTheta, X, Y, myLamda=1.))

def computeGradient(myTheta, myX, myY, myLamda=.0):
    myTheta = myTheta.reshape((myTheta.shape[0],1))
    m =  myX.shape[0]

    myH = h(myTheta, myX)
    grad = float(1/m) * (myH - myY).T.dot(myX)

    regTerm = float(myLamda/m) * myTheta

    regTerm = np.delete(regTerm, 0) # Don't regulate bias unit

    return (grad + regTerm)[0]

# print(computeGradient(myTheta, X, Y, 1))

def optimizeTheta(myTheta_initial, myX, myY, myLamda=.0, print_output = True):
    fit_theta = scipy.optimize.fmin_cg(computeCost, 
                                        x0=myTheta_initial, 
                                        fprime=computeGradient, 
                                        args=(myX, myY, myLamda),
                                        disp=print_output, 
                                        epsilon=1e-5, 
                                        maxiter=1000)
    fit_theta = fit_theta.reshape((myTheta_initial.shape[0],1))

    return fit_theta

fit_theta = optimizeTheta(myTheta, X, Y, 0.)

print(fit_theta)

# DRAW TWO FIGURES AT THE SAME WINDOW
# plotData()
# plt.plot(X[:,1], h(fit_theta,X).flatten())
# plt.show()

# this function is running with a error, not urgent,
# fix it as soon as you're about to complete assignment 5
def plotLearningCurve():
    """
    Loop over first training point, then first 2 training points, then first 3 ...
    and use each training-set-subset to find trained parameters.
    With those parameters, compute the cost on that subset (Jtrain)
    remembering that for Jtrain, lambda = 0 (even if you are using regularization).
    Then, use the trained parameters to compute Jval on the entire validation set
    again forcing lambda = 0 even if using regularization.
    Store the computed errors, error_train and error_val and plot them.
    """
    initial_theta = np.array([[1.],[1.]])
    myM, error_training, error_val = [], [], []
    for x in range(1,13,1):
        train_subset = X[:x,:]
        y_subset = Y[:x]
        myM.append(y_subset.shape[0])
        fit_theta = optimizeTheta(initial_theta, train_subset, y_subset, myLamda=.0, print_output=False)
        error_training.append(computeCost(fit_theta, train_subset, y_subset, myLamda=.0))
        error_val.append(computeCost(fit_theta, Xval, Yval, myLamda=.0))

    plt.figure(figsize=(8,5))
    plt.plot(myM, error_training, label='Traing')
    plt.plot(myM, error_val, label='Cross Validation')
    plt.legend()
    plt.title('Polynomial Regression Learning Curve (lambda = 0)')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.grid(True)

# plotLearningCurve()
# plt.show()
    
def genPolyFeatures(myX, p):
    """
    Function takes in the X matrix (with bias term already included as the first column)
    and returns an X matrix with "p" additional columns.
    The first additional column will be the 2nd column (first non-bias column) squared,
    the next additional column will be the 2nd column cubed, etc.
    """
    newX = myX.copy()

    for i in range(p):
        dim = i + 2
        newX = np.insert(newX, newX.shape[1], np.power(newX[:,1], dim), axis=1)

    return newX

def featuresNormalize(myX):
    """
    Takes as input the X array (with bias "1" first column), does
    feature normalizing on the columns (subtract mean, divide by standard deviation).
    Returns the feature-normalized X, and feature means and stds in a list
    """

    Xnorm = myX.copy()
    stored_feature_means = np.mean(Xnorm, axis=0) # column by column
    Xnorm[:,1:] = Xnorm[:,1:] - stored_feature_means[1:]
    stored_feature_stds = np.std(Xnorm, axis=0, ddof=1)
    Xnorm[:,1:] = Xnorm[:,1:] / stored_feature_stds[1:]

    return Xnorm, stored_feature_means, stored_feature_stds

global_d = 5
newX = genPolyFeatures(X, global_d)
newX_norm, stored_means, stored_stds = featuresNormalize(newX)

_myTheta = np.ones((newX_norm.shape[1],1))
fit_theta = optimizeTheta(_myTheta, newX_norm, Y, myLamda=.0)