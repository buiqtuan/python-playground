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
from sklearn import svm
import time

ex6data1 = './DataAssignment6/ex6data1.mat'
ex6data2 = './DataAssignment6/ex6data2.mat'
ex6data3 = './DataAssignment6/ex6data3.mat'

# Check if in debug mode
gettrace = getattr(sys, 'gettrace', None)

if gettrace():
    print('In Debug Mode!')
    ex6data1 = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment6\ex6data1.mat'
    ex6data2 = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment6\ex6data2.mat'
    ex6data3 = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment6\ex6data3.mat'

mat1 = io.loadmat(ex6data1)
mat2 = io.loadmat(ex6data2)
mat3 = io.loadmat(ex6data3)

# # Training Set
X, Y = mat1['X'], mat1['y']

# # NOT inserting a column of 1's in case SVM software does it automatically...
# # X = np.insert(X, 0, 1, axis=1)

# # Devide data into 2 part, negative and positive
pos = np.array([X[i] for i in range(X.shape[0]) if Y[i] == 1])
neg = np.array([X[i] for i in range(X.shape[0]) if Y[i] == 0])

def plotData():
    plt.figure(figsize=(10,6))
    plt.plot(pos[:,0], pos[:,1], 'k+', label = 'Positive Sample')
    plt.plot(neg[:,0], neg[:,1], 'yo', label = 'Negative Sample')
    plt.xlabel('Column 1 Variables')
    plt.ylabel('Column 2 Variables')
    plt.legend()
    plt.grid(True)

# Function to plot the SVM decision boundary
def plotSVMBoundary(mySVM, Xmin, Xmax, Ymin, Ymax):
    """
    Function to plot the decision boundary for a trained SVM
    It works by making a grid of x1 ("xvals") and x2 ("yvals") points,
    And for each, computing whether the SVM classifies that point as
    True or False. Then, a contour is drawn with a built-in pyplot function.
    """
    xvals = np.linspace(Xmin, Xmax, 500)
    yvals = np.linspace(Ymin, Ymax, 500)
    zvals = np.zeros((len(xvals), len(yvals)))
    for i in range(len(xvals)):
        for j in range(len(yvals)):
            zvals[i][j] = float(mySVM.predict(np.array([[xvals[i],yvals[j]]])))
    
    zvals = zvals.transpose()

    u,v = np.meshgrid(xvals, yvals)
    myContour = plt.contour(xvals, yvals, zvals, [0])
    plt.title('Decision Boundary')

# Run the SVM training (with C = 1) using SVM software. 
# When C = 1, you should find that the SVM puts the decision boundary 
# in the gap between the two datasets and misclassifies the data point on the far left

# Make an instance of an SVM with C=1 and 'linear' kernel
linear_svm = svm.SVC(C=100, kernel='linear') # Create a SVM model, C = C value, linear is linear Kernel, gamma = 0
# Fit the SVM to our X matrix (no bias unit)
linear_svm.fit(X, Y.flatten())
# Plot the decision boundary
# plotData(pos, neg)
# plotSVMBoundary(linear_svm, 0, 4.5, 1.5, 5)
# plt.show()

# Here's how to use this SVM software with a custom kernel:
# http://scikit-learn.org/stable/auto_examples/svm/plot_custom_kernel.html
def gaussKernel(x1, x2, sigma):
    sigmaSquared = np.power(sigma, 2)
    kernel = -(x1 - x2).T.dot(x1 - x2)
    return np.exp(kernel/(2*sigmaSquared))

# x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
# sim = gaussianKernel(x1, x2, sigma);
# print(gaussKernel(np.array([1, 2, 1]),np.array([0, 4, -1]), 2.)) # this value should be about 0.324652 / check my implementation of gaussian kernel

# Now that I've shown I can implement a gaussian Kernel,
# I will use the of-course built-in gaussian kernel in my SVM software
# because it's certainly more optimized than mine.
# It is called 'rbf' and instead of dividing by sigmasquared,
# it multiplies by 'gamma'. As long as I set gamma = sigma^(-2),
# it will work just the same.

X, Y = mat2['X'], mat2['y']

# Devide data into 2 part, negative and positive
pos = np.array([X[i] for i in range(X.shape[0]) if Y[i] == 1])
neg = np.array([X[i] for i in range(X.shape[0]) if Y[i] == 0])

# plotData(pos, neg)
# plt.show()

# Train the SVM with the Gaussian kernel on this dataset.
# sigma = .1
# gamma = sigma**-2

# started_time = int(round(time.time() * 1000))
# gaus_svm = svm.SVC(C=1, kernel='rbf', gamma=gamma) # Create a SVM model, C = C value, rbf is Gaussian Kernel, gamma = gamma value
# gaus_svm.fit(X, Y.flatten()) # in this example, using flattened numpy array is slightly faster than normal numpy array (in 4 out of 5 times runing this)
# plotData()
# plotSVMBoundary(gaus_svm, 0, 1, .4, 1.)
# print("Done training in: %i milliseconds"%(int(round(time.time() * 1000)) - started_time))
# plt.show()

X, Y = mat3['X'], mat3['y']
Xval, Yval = mat3['Xval'], mat3['yval']

# Devide data into 2 part, negative and positive
pos = np.array([X[i] for i in range(X.shape[0]) if Y[i] == 1])
neg = np.array([X[i] for i in range(X.shape[0]) if Y[i] == 0])

# Your task is to use the cross validation set Xval, yval to 
# determine the best C and σ parameter to use.

# The score() function for a trained SVM takes in
# X and y to test the score on, and the (float)
# value returned is "Mean accuracy of self.predict(X) wrt. y"

Cvalues = SigmaValues = (0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.)



def calculateBestScore(_Cvalues, _SigmaValues):
    best_pair, best_score = (0, 0), 0

    for C in _Cvalues:
        for sigma in _SigmaValues:
            gamma = sigma**-2
            gaus_svm = svm.SVC(C,kernel='rbf',gamma=gamma) # Create a SVM model, C = C value, rbf is Gaussian Kernel, gamma = gamma value
            gaus_svm.fit(X, Y.flatten()) 
            this_score = gaus_svm.score(Xval, Yval) # Evaluate the score base on the cross validation set
            if (this_score > best_score):
                best_score = this_score
                best_pair = (C, sigma)

    return (best_pair, best_score)

bestValues = calculateBestScore(Cvalues, SigmaValues)

gaus_svm = svm.SVC(C=bestValues[0][0],
                    kernel='rbf',
                    gamma=(bestValues[0][1]**-2))
gaus_svm.fit(X, Y.flatten())
plotData()
plotSVMBoundary(gaus_svm, -.5, .3, -.8, .6)
plt.show()