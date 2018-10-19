from __future__ import print_function

import sys
import random
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import io
from scipy import linalg

birdSmallPath = './DataAssignment7/bird_small.mat'
ex7data1Path = './DataAssignment7/ex7data1.mat'
ex7data2Path = './DataAssignment7/ex7data2.mat'
ex7data3Path = './DataAssignment7/ex7data3.mat'

# Check if in debug mode
gettrace = getattr(sys, 'gettrace', None)

if (gettrace()):
	birdSmallPath = '.D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment7\bird_small.mat'
	ex7data1Path = '.D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment7\ex7data1.mat'
	ex7data2Path = '.D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment7\ex7data2.mat'
	ex7data3Path = '.D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment7\ex7data3.mat'

# load data from mat file
ex7data2 = io.loadmat(ex7data2Path)
X = ex7data2['X']

print(X.shape)

# Choose the number of centroids with K = 3
K = 3
# Choose the initial centroids matching the data
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Visualize the data
def plotData(myX: np.array, myCentroids: list, myIdxs: np.array = None):
	"""
	Fucntion to plot the data and color it accordingly.
    myIdxs should be the latest iteraction index vector
    myCentroids should be a vector of centroids, one per iteration
	"""
	colors = ['b','g','gold','darkorange','salmon','olivedrab']

	assert myX[0].shape == myCentroids[0][0].shape
	assert myCentroids[-1].shape[0] <= len(colors)

	subX = []
	# If idxs is supplied, divide up X into colors
	if myIdxs is not None:
		assert myIdxs.shape[0] == myX.shape[0]
		
		for x in range(myCentroids[0].shape[0]):
			subX.append(np.array([myX[i] for i in range(myX.shape[0]) if myIdxs[i] == x]))
	else:
		subX = [myX]
	
	fig = plt.figure(figsize=(7,5))
	newX = None
	for x in range(len(subX)):
		newX = subX[x]
		plt.plot(newX[:,0], newX[:,1], 'o', color=colors[x], alpha=0.75, label='Data Points: Cluster %d'%x)
	
	plt.xlabel('x1', fontsize=14)
	plt.ylabel('x2', fontsize=14)
	plt.title('Plot of X points', fontsize=16)
	plt.grid(True)

	# Drawing a history of centroid movement
	tempx, tempy = [], []
	for myC in myCentroids:
		tempx.append(myC[:,0])
		tempy.append(myC[:,1])

	for x in range(len(tempx[0])):
		plt.plot(tempx, tempy, 'rx--', marketsize=8)

	plt.legend(loc=4, framealpha=0.5)

plotData(X, [initial_centroids])
plt.show()