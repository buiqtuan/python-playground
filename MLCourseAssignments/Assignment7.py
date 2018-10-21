from __future__ import print_function

import sys
from random import sample
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import io
from scipy import linalg
import imageio

birdMatPath = './DataAssignment7/bird_small.mat'
birdImgPath = './DataAssignment7/bird_small.png'
ex7data1Path = './DataAssignment7/ex7data1.mat'
ex7data2Path = './DataAssignment7/ex7data2.mat'
ex7data3Path = './DataAssignment7/ex7faces.mat'

# Check if in debug mode
gettrace = getattr(sys, 'gettrace', None)

if (gettrace()):
	birdMatPath = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment7//bird_small.mat'
	birdImgPath = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment7//bird_small.png'
	ex7data1Path = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment7\ex7data1.mat'
	ex7data2Path = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment7\ex7data2.mat'
	ex7data3Path = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment7\ex7faces.mat'

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
	
	plt.figure(figsize=(7,7))
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
		plt.plot(tempx, tempy, 'rx--')

	plt.legend(loc=4, framealpha=0.5)

# plotData(X, [initial_centroids])
# plt.show()

def distSquared(point1: np.array, point2: np.array):
	assert point1.shape == point2.shape
	return np.sum(np.square(point2-point1))

def findClosesetCentroids(myX, myCentroids):
	"""
	Function takes in the (m,n) X matrix
    (where m is the # of points, n is # of features per point)
    and the (K,n) centroid seed matrix
    (where K is the # of centroids (clusters))
    and returns a (m,1) vector of cluster indices 
    per point in X (0 through K-1)
	"""
	idxs = np.zeros((myX.shape[0],1))

	# Loop through each data point in X
	for x in range(idxs.shape[0]):
		myPoint = myX[x]
		#Compare this point to each centroid,
        #Keep track of shortest distance and index of shortest distance
		minDist, idx = sys.maxsize, 0
		for i in range(myCentroids.shape[0]):
			myC = myCentroids[i]
			dist = distSquared(myC, myPoint)
			if (dist < minDist):
				minDist = dist
				idx = i
		
		#With the best index found, modify the result idx vector
		idxs[x] = idx
	
	return idxs

# idxs = findClosesetCentroids(X, initial_centroids)

# print(idxs[:3].flatten())

# plotData(X, [initial_centroids], idxs)
# plt.show()

def computeMeanCentroids(myX, myIdsx):
	"""
	Function takes in the X matrix and the index vector
    and computes a new centroid matrix.
	"""
	subX = []
	for x in range(len(np.lib.arraysetops.unique(myIdsx))):
		# Create an array for points that are closer to a centroid, then append to subX
		subX.append(np.array([myX[i] for i in range(myX.shape[0]) if myIdsx[i] == x]))
	
	return np.array([np.mean(thisX, axis=0) for thisX in subX])

def runKMeans(myX, initial_centroids, K, n_iter):
	"""
	Function that actually does the iterations
	"""
	centroid_history = []
	current_centrois = initial_centroids
	for i in range(n_iter):
		centroid_history.append(current_centrois)
		idxs = findClosesetCentroids(myX, current_centrois)
		current_centrois = computeMeanCentroids(myX, idxs)

	return idxs, centroid_history

# idxs, centroid_history = runKMeans(X, initial_centroids, K=3,n_iter=10)

# plotData(X, centroid_history, idxs)
# plt.show()

def chooseKRandomCentroids(myX, K):
	rand_indices = sample(range(0, myX.shape[0]), K)
	return np.array([myX[i] for i in rand_indices])

# for x in range(3):
# 	idxs, centroid_history = runKMeans(X, chooseKRandomCentroids(X, K=3), K=3, n_iter=10)

# 	plotData(X, centroid_history, idxs)
# 	plt.show()

birdImg = imageio.imread(birdImgPath)
print("BirdImg shape is: ", birdImg.shape)

# plt.imshow(birdImg)
# plt.show()

# Divide every entry in A by 255 so all values are in the range of 0 to 1
birdImg = birdImg / 255

# Unroll the image to shape (16384,3) (16384 is 128*128)
birdImg = birdImg.reshape(-1, 3)

# Run k-means on this data, forming 16 clusters, with random initialization
myK = 16
idxs, centroid_history = runKMeans(birdImg, chooseKRandomCentroids(birdImg, myK), myK, n_iter=10)

# Now I have 16 centroids, each representing a color.
# Let's assign an index to each pixel in the original image dictating
# which of the 16 colors it should be
idxs = findClosesetCentroids(birdImg, centroid_history[-1])

final_centroids = centroid_history[-1]
# Now loop through the original image and form a new image
# that only has 16 colors in it
final_image = np.zeros((idxs.shape[0],3))
for x in range(final_image.shape[0]):
	final_image[x] = final_centroids[int(idxs[x])]

# Reshape the original image and the new, final image and draw them
# To see what the "compressed" image looks like
plt.figure()
dummy = plt.imshow(birdImg.reshape(128,128,3))
plt.show()
plt.figure()
dummy = plt.imshow(final_image.reshape(128,128,3))
plt.show()