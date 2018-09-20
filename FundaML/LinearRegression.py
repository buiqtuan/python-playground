from __future__ import print_function
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn import datasets, linear_model

# height (cm), input data, each row is a data point
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# Linear regression using normal equation h = (X.T * X)^-1 * X.T * Y
# Building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one,X), axis = 1)

# Calculating weights of the fitting line
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A),b)

# Weights
w_0, w_1 = w[0], w[1]

y1 = w_1*155 + w_0
y2 = w_1*160 + w_0
print('Input 155cm, true output 52kg, predicted output %.2fkg' %(y1) )
print('Input 160cm, true output 56kg, predicted output %.2fkg' %(y2) )

# Using scikit-learn
# fit the model by Linear Regression
regr = linear_model.LinearRegression()
regr.fit(X,y)

# Compare two results
print("scikit-learn’s solution : w_1 = ", regr.coef_[0], "w_0 = ", regr.intercept_)
print("our solution : w_1 = ", w[1], "w_0 = ", w[0])