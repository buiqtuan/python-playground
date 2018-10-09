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