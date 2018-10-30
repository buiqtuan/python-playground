from __future__ import print_function

import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.optimize as optimize
from scipy import io

ex8data1Path = './DataAssignment8/ex8data1.mat'

# Check if in debug mode
gettrace = getattr(sys, 'gettrace', None)

if (gettrace()):
    print('In debug Mode!')
    ex8data1Path = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment8\ex8data1.mat'

ex8data1 = io.loadmat(ex8data1Path)
X = ex8data1['X']
Ycv = ex8data1['yval']
Xcv = ex8data1['Xval']

# Visualize the data
def plotData(myX, newFig=False):
    if (newFig):
        plt.figure(figsize=(8,6))
    
    plt.plot(myX[:,0],myX[:,1],'b+')
    plt.xlabel('Latency [ms]', fontsize=16)
    plt.ylabel('Throughput [mb/s]', fontsize=16)
    plt.grid(True)

# plotData(X, False)
# plt.show()

def gaus(myX, myMu, mySig2):
    """
    Function to compute the gaussian return values for a feature
    matrix, myX, given the already computed mu vector and sigma matrix.
    If sigma is a vector, it is turned into a diagonal matrix
    Uses a loop over rows; didn't quite figure out a vectorized implementation.
    """
    m = myX.shape[0]
    n = myX.shape[1]
    if (np.ndim(mySig2) == 1):
        mySig2 = np.diag(mySig2)

    # np.linalg.det: calculate determinant of a matrix
    # To calculate the multiplication of all element in sigma vector:
    # 1. np.diag (diagnosis a matrix)
    # 2. compute its determinant
    norm = 1/(np.power((2*np.pi), n/2)*np.sqrt(np.linalg.det(mySig2)))
    # The (1/2*sigma) part in Gaussian Distribution formular can be calculated by finding its inverse matrix
    myinv = np.linalg.inv(mySig2) 
    myexp = np.zeros((m,1))

    for i in range(m):
        xrow = myX[i]
        myexp[i] = np.exp(-0.5*((xrow-myMu).T).dot(myinv).dot(xrow-myMu))

    return norm*myexp

def getGaussianParams(myX, useMultivariate=True):
    """
    Function that given a feature matrix X that is (m x n)
    returns a mean vector and a sigmasquared vector that are
    both (n x 1) in shape.
    This can do it either as a 1D gaussian for each feature,
    or as a multivariate gaussian.
    """
    m = myX.shape[0]
    # compute means of n features
    mu = np.mean(myX, axis=0)

    if not useMultivariate:
        sigma2 = np.sum(np.square(myX - mu), axis=0)/float(m)
        return mu, sigma2
    else:
        # useMultivariate=True
        sigma2 = ((myX - mu).T.dot(myX - mu))/float(m)
        return mu, sigma2

mu, sig2 = getGaussianParams(X, useMultivariate=True)
# print(mu, sig2)

# Visualizing the Gaussian probability contours
def plotContours(myMu, mySigma2, newFig=False):
    delta = .05
    myX = np.arange(0,30,delta)
    myY = np.arange(0,30,delta)
    meshX, meshY = np.meshgrid(myX, myY)
    coord_list = [ entry.ravel() for entry in (meshX, meshY) ]
    points = np.vstack(coord_list).T
    myZ = gaus(points, myMu, mySigma2)

    myZ = myZ.reshape((myX.shape[0], myX.shape[0]))

    if (newFig):
        plt.figure(figsize=(6,4))

    cont_levels = [10**exp for exp in range(-20,0,3)]
    myCont = plt.contour(meshX, meshY, myZ, levels=cont_levels)

    plt.title('Gaussian Contours', fontsize=16)

# Using multivariate Gauss
# plotData(X, newFig=True)
# plotContours(*getGaussianParams(X,useMultivariate=True), newFig=False)
# plt.show()

# Not Using multivariate Gauss
# plotData(X, newFig=True)
# plotContours(*getGaussianParams(X,useMultivariate=False), newFig=False)
# plt.show()