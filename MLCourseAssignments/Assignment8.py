from __future__ import print_function

import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.optimize as optimize
from scipy import io

ex8data1Path = './DataAssignment8/ex8data1.mat'
ex8data2Path = './DataAssignment8/ex8data2.mat'

# Check if in debug mode
gettrace = getattr(sys, 'gettrace', None)

if (gettrace()):
    print('In debug Mode!')
    ex8data1Path = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment8\ex8data1.mat'
    ex8data2Path = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment8\ex8data2.mat'

ex8data1 = io.loadmat(ex8data1Path)
X = ex8data1['X']
Ycv = ex8data1['yval']
Xcv = ex8data1['Xval']

ex8data2 = io.loadmat(ex8data2Path)
X2 = ex8data2['X']
Y2cv = ex8data2['yval']
X2cv = ex8data2['Xval']

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

# mu, sig2 = getGaussianParams(X, useMultivariate=True)
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

def computeF1(predVec, trueVec):
    """
    F1 = 2 * (P*R)/(P+R)
    where P is precision, R is recall
    Precision = "of all predicted y=1, what fraction had true y=1"
    Recall = "of all true y=1, what fraction predicted y=1?
    Note predictionVec and trueLabelVec should be boolean vectors.
    """
    P, R = 0., 0.
    if (float(np.sum(predVec))):
        P = np.sum([int(trueVec[x]) for x in range(predVec.shape[0]) if predVec[x]]) / float(np.sum(predVec))

    if (float(np.sum(trueVec))):
        R = np.sum([int(predVec[x]) for x in range(trueVec.shape[0]) if trueVec[x]]) / float(np.sum(trueVec))

    return 2*P*R/(P+R) if (P+R) else 0

def selectThreshold(myYCV, myPCVs):
    """
    Function to select the best epsilon value from the CV set
    by looping over possible epsilon values and computing the F1
    score for each.
    """
    # Make a list of possible epsilon values
    nsteps = 1000
    epses = np.linspace(np.min(myPCVs), np.max(myPCVs), nsteps)

    # Compute the F1 score for each epsilon value, and store the best 
    # F1 score (and corresponding best epsilon)
    bestF1, bestEps = 0, 0
    trueVec = (myYCV == 1).flatten()
    for eps in epses:
        predVec = myPCVs < eps
        thisF1 = computeF1(predVec, trueVec)
        if (thisF1 > bestF1):
            bestF1 = thisF1
            bestEps = eps

    print("Best F1 is %f, best eps is %0.4g."%(bestF1,bestEps))
    return bestF1, bestEps

# Using the gaussian parameters from the full training set,
# figure out the p-value for each point in the CV set
# pCVs = gaus(Xcv, mu, sig2)

# bestF1, bestEps = selectThreshold(Ycv, pCVs)

def plotAnomalies(myX, myBestEsp, newFig=False, useMultivariate=True):
    ps = gaus(myX, *getGaussianParams(myX, useMultivariate))
    anoms = np.array([myX[x] for x in range(myX.shape[0]) if ps[x] < myBestEsp])
    if (newFig):
        plt.figure(figsize=(6,4))
    plt.scatter(anoms[:,0], anoms[:,1], s=80, facecolors='none', edgecolors='r')

# plotData(X, newFig=True)
# plotContours(mu, sig2, newFig=False)
# plotAnomalies(X, bestEps, newFig=False, useMultivariate=True)
# plt.show()

# Experiment on larger dataset with 11 features
mu, sig2 = getGaussianParams(X2, useMultivariate=False)
ps = gaus(X2, mu, sig2)
pCVs = gaus(X2cv, mu, sig2)

# Using the gaussian parameters from the full training set,
# figure out the p-value for each point in the CV set

bestF1, bestEps = selectThreshold(Y2cv, pCVs)
anoms = [X2[x] for x in range(X2.shape[0]) if ps[x] < bestEps]

print('# of anomalies found: ',len(anoms))
