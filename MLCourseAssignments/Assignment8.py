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
    norm = 1/(np.power((2*np.pi), n/2)*np.sqrt(np.linalg.det(mySig2)))
    myinv = np.linalg.inv(mySig2)
    myexp = np.zeros((m,1))

    for i in range(m):
        xrow = myX[i]
        myexp[i] = np.exp(-0.5*((xrow-mymu).T).dot(myinv).dot(xrow-mymu))

    return norm*myexp