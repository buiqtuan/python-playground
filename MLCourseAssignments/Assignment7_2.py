from __future__ import print_function

import PIL
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
ex7data1Path = './DataAssignment7/ex7data1.mat'
ex7facesPath = './DataAssignment7/ex7faces.mat'
birdImgPath = './DataAssignment7/bird_small.png'

# Check if in debug mode
gettrace = getattr(sys, 'gettrace', None)

if (gettrace()):
    print('In debug Mode!')
    birdMatPath = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment7//bird_small.mat'
    ex7data1Path = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment7\ex7data1.mat'
    ex7data3Path = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment7\ex7faces.mat'
    birdImgPath = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment7//bird_small.png'

mat1 = io.loadmat(ex7data1Path)

X = mat1['X']

def plotData():
    plt.figure(figsize=(7,7))
    plt.plot(X[:,0], X[:,1], 'o')
    plt.title('Example Dataset')
    plt.grid(True)

def featureNormalize(myX):
    # Feature-normalize X, return it
    means = np.mean(myX, axis=0)
    myX_norm = myX - means
    stds = np.std(myX, axis=0)
    myX_norm = myX_norm / stds
    return means, stds, myX_norm

def getUSV(myX_norm):
    # Compute the covarience matrix
    cov_matrix = myX_norm.T.dot(myX_norm)/myX_norm.shape[0]
    # Run single value decomposition to get the U principal component matrix
    U, S, V = scipy.linalg.svd(cov_matrix, full_matrices=True, compute_uv=True)
    return U, S, V

# Feature normalize
# means, stds, X_norm = featureNormalize(X)
# Run SVD
# U, S, V = getUSV(X_norm)

# print('Top principal component is ',U[:,0])

def plotPCA(myX, means, S, U):
    plt.figure(figsize=(7,7))
    plt.scatter(X[:,0], X[:,1], s=30, facecolors='none', edgecolors='b')
    plt.title("Example Dataset: PCA Eigenvectors Shown",fontsize=18)
    plt.xlabel('x1',fontsize=18)
    plt.ylabel('x2',fontsize=18)
    plt.grid(True)
    # To draw the principal component, draw them starting
    # at the mean of the data
    plt.plot([means[0], means[0] + 1.5*S[0]*U[0,0]], 
            [means[1], means[1] + 1.5*S[0]*U[0,1]],
            color='red',linewidth=3,
            label='First Principal Component')
    plt.plot([means[0], means[0] + 1.5*S[1]*U[1,0]], 
            [means[1], means[1] + 1.5*S[1]*U[1,1]],
            color='fuchsia',linewidth=3,
            label='Second Principal Component')
    plt.legend(loc=4)

# plotPCA(X, means, S, U)
# plt.show()

def projectData(myX, myU, K):
    """
    Function that computes the reduced data representation when
    projecting only on to the top "K" eigenvectors
    """
    Ureduced = myU[:,:K]
    z = myX.dot(Ureduced)
    return z

# z = projectData(X_norm, U, 1)
# print('Projection of the first example is %0.3f.'%float(z[0]))

def recoverData(myZ, myU, K):
    Ureduced = U[:,:K]
    Xapprox = myZ.dot(Ureduced.T)
    return Xapprox

# X_rec = recoverData(z, U, 1)
# print('Recovered approximation of the first example is ',X_rec[0])

def plotProjectedLine(myX_norm, myX):
    
    #Quick plot, now drawing projected points to the original points
    plt.figure(figsize=(7,5))
    plot = plt.scatter(myX_norm[:,0], myX_norm[:,1], s=30, facecolors='none', 
                    edgecolors='b',label='Original Data Points')
    plot = plt.scatter(myX[:,0], myX[:,1], s=30, facecolors='none', 
                    edgecolors='r',label='PCA Reduced Data Points')

    plt.title("Example Dataset: Reduced Dimension Points Shown",fontsize=14)
    plt.xlabel('x1 [Feature Normalized]',fontsize=14)
    plt.ylabel('x2 [Feature Normalized]',fontsize=14)
    plt.grid(True)

    for x in range(myX_norm.shape[0]):
        plt.plot([myX_norm[x,0],myX[x,0]],[myX_norm[x,1],myX[x,1]],'k--')
        
    leg = plt.legend(loc=4)

    #Force square axes to make projections look better
    dummy = plt.xlim((-2.5,2.5))
    dummy = plt.ylim((-2.5,2.5))

# plotProjectedLine(X_norm, X_rec)
# plt.show()

matFaces = io.loadmat(ex7facesPath)

faces = matFaces['X']

def getDatumImg(row):
    """
    Function that is handed a single np array with shape 1x1032,
    crates an image object from it, and returns it
    """
    width, height = 32, 32
    square = row.reshape(width, height)
    return square.T

def displayData(myX, myNrows=10, myNcols=10):
    """
    Function that picks the first 100 rows from X, creates an image from each,
    then stitches them together into a 10x10 grid of images, and shows it.
    """
    width, height = 32, 32
    nrows, ncols = myNrows, myNcols

    big_picture = np.zeros((height*nrows, width*ncols))

    irow, icol = 0, 0
    for i in range(nrows*ncols):
        if (icol == ncols):
            irow += 1
            icol = 0
        iimg = getDatumImg(myX[i])
        big_picture[irow*height:irow*height + iimg.shape[0],icol*width:icol*width + iimg.shape[1]] = iimg
        icol += 1
    
    fig = plt.figure(figsize=(10,10))
    img = PIL.Image.fromarray(big_picture)
    plt.imshow(img, cmap='Greys_r')

# displayData(faces)
# plt.show()

# Doing feature normalize on faces data
# means, stds, X_norm = featureNormalize(faces)
# Run SVD
# U, S, V = getUSV(X_norm)

# Visualize the top 36 eigenvectors found
# displayData(U[:,:36].T, myNrows=6, myNcols=6)
# plt.show()

# Project each image down to 36 dimensions
# z = projectData(X_norm, U, K=36)

# Attempt to recover the original data
# X_rec = recoverData(z, U, K=36)

# Plot the dimension-reduced data
# displayData(X_rec)
# plt.show()

# read birdImg into numpy array A
A = imageio.imread(birdImgPath)

# Divide every entry in A by 255 so all values are in the range of 0 to 1
A = A / 255

# Unroll the image to shape (16384,3) (16384 is 128*128)
A = A.reshape(-1, 3)

means, stds, A_norm = featureNormalize(A)

U, S, V = getUSV(A_norm)

# Use PCA  to transform from 3 to 2 dimensions
z = projectData(A_norm, U, K=2)

# def plot2DData():
#     # optional: in progress