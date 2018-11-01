from __future__ import print_function

import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.optimize as optimize
from scipy import io

ex8moviesPath = './DataAssignment8/ex8_movies.mat'
ex8movieParamsPath = './DataAssignment8/ex8_movieParams.mat'
movie_idsPath = './DataAssignment8/movie_ids.txt'

np.random.seed(7)
# Check if in debug mode
gettrace = getattr(sys, 'gettrace', None)

if (gettrace()):
    print('In debug Mode!')
    ex8moviesPath = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment8\ex8_movies.mat'
    ex8movieParamsPath = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment8\ex8_movieParams.mat'
    movie_idsPath = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment8\movie_ids.txt'

ex8_movies = io.loadmat(ex8moviesPath)
Y = ex8_movies['Y']
R = ex8_movies['R']

nm, nu = Y.shape
# Y is 1682x943 containing ratings (1-5) of 1682 movies on 943 users
# a rating of 0 means the movie wasn't rated
# R is 1682x943 containing R(i,j) = 1 if user j gave a rating to movie i
# print('Average rating for movie 1 (Toy Story) is: %0.2f'%np.mean([ Y[0][x] for x in range(Y.shape[1]) if R[0][x] ]))

ex8_movieParams = io.loadmat(ex8movieParamsPath)
X = ex8_movieParams['X']
Theta = ex8_movieParams['Theta']
nu = int(ex8_movieParams['num_users'])
nm = int(ex8_movieParams['num_movies'])
nf = int(ex8_movieParams['num_features'])

# For now, reduce the data set size so that this runs faster
nu = 4; nm = 5; nf = 3
X = X[:nm,:nf]
Theta = Theta[:nu,:nf]
Y = Y[:nm,:nu]
R = R[:nm,:nu]

movies = []
with open(movie_idsPath) as f:
    for line in f:
        movies.append(' '.join(line.strip('\n').split(' ')[1:]))

# "Visualize the ratings matrix"
def plotRatingMatrix(myY):
    plt.figure(figsize=(6,6*(myY.shape[0]/myY.shape[1])))
    plt.imshow(myY)
    plt.colorbar()
    plt.ylabel('Movies (%d)'%nm,fontsize=20)
    plt.xlabel('Users (%d)'%nu,fontsize=20)

# plotRatingMatrix(Y)
# plt.show()


# Throughout this part of the exercise, you will also be 
# working with the matrices, X and Theta
# The i-th row of X corresponds to the feature vector x(i) for the i-th movie, 
# and the j-th row of Theta corresponds to one parameter vector θ(j), for the j-th user. 
# Both x(i) and θ(j) are n-dimensional vectors. For the purposes of this exercise, 
# you will use n = 100, and therefore, x(i) ∈ R100 and θ(j) ∈ R100. Correspondingly, 
# X is a nm × 100 matrix and Theta is a nu × 100 matrix.

# The "parameters" we are minimizing are both the elements of the
# X matrix (nm*nf) and of the Theta matrix (nu*nf)
# To use off-the-shelf minimizers we need to flatten these matrices
# into one long array

def flattenParams(myX, myTheta):
    """
    Hand this function an X matrix and a Theta matrix and it will flatten
    it into into one long (nm*nf + nu*nf,1) shaped numpy array
    """
    return np.concatenate((myX.flatten(), myTheta.flatten()))

# A utility function to re-shape the X and Theta will probably come in handy
def reshapeParams(flattened_XandTheta, myNM, myNU, myNF):
    assert flattened_XandTheta.shape[0] == int(myNM*myNF + myNU*myNF)

    reX = flattened_XandTheta[:int(myNM*myNF)].reshape((myNM, myNF))
    reTheta = flattened_XandTheta[int(myNM*myNF):].reshape((myNU, myNF))
    
    return reX, reTheta

def cofiCostFunc(myParams, myY, myR, myNU, myNM, myNF, myLambda=0.):

    # Unfold the X and Theta matrices from the flattened params
    myX, myTheta = reshapeParams(myParams, myNM, myNU, myNF)

    # Note: 
    # X Shape is (nm x nf), Theta shape is (nu x nf), Y and R shape is (nm x nu)
    # Complete vectorization

    # First dot theta and X together such that you get a matrix the same shape as Y
    term1 = myX.dot(myTheta.T)

    # Then element-wise multiply that matrix by the R matrix
    # so only terms from movies which that user rated are counted in the cost
    term1 = np.multiply(term1, myR)

    # Then subtract the Y- matrix (which has 0 entries for non-rated
    # movies by each user, so no need to multiply that by myR... though, if
    # a user could rate a movie "0 stars" then myY would have to be element-
    # wise multiplied by myR as well) 
    # also square that whole term, sum all elements in the resulting matrix,
    # and multiply by 0.5 to get the cost
    cost = 0.5*np.sum(np.square(term1 - myY))

    # regularization stuff
    cost += (myLambda/2.) * np.sum(np.square(myTheta))
    cost += (myLambda/2.) * np.sum(np.square(myX))

    return cost

# print('Cost with nu = 4, nm = 5, nf = 3 is %0.2f.' %cofiCostFunc(flattenParams(X,Theta),Y,R,nu,nm,nf))

# print('Cost with nu = 4, nm = 5, nf = 3 is %0.2f.' %cofiCostFunc(flattenParams(X,Theta),Y,R,nu,nm,nf, myLambda=1.5))

# Remember: use the exact same input arguments for gradient function
# as for the cost function (the off-the-shelf minimizer requires this)
def cofiGrad(myParams, myY, myR, myNU, myNM, myNF, myLambda=0.):

    # Unfold the X and Theta matrices from the flattened params
    myX, myTheta = reshapeParams(myParams, myNM, myNU, myNF)

    # First the X gradient term 
    # First dot theta and X together such that you get a matrix the same shape as Y
    term1 = myX.dot(myTheta.T)

    # Then multiply this term by myR to remove any components from movies that
    # weren't rated by that user
    term1 = np.multiply(term1, myR)

    # Now subtract the y matrix (which already has 0 for nonrated movies)
    term1 -= myY

    # Lastly dot this with Theta such that the resulting matrix has the
    # same shape as the X matrix
    XGrad = term1.dot(myTheta)

    # Now the Theta gradient term (reusing the "term1" variable)
    ThetaGrad = term1.T.dot(myX)

    # Regularization stuff
    XGrad += myLambda*myX
    ThetaGrad += myLambda*myTheta

    return flattenParams(XGrad, ThetaGrad)

def checkGradient(myParams, myY, myR, myNU, myNM, myNF, myLambda=0.):
    print('Numerical Gradient \t cofiGrad \t\t Difference')

    # Compute a numerical gradient with an epsilon perturbation vector
    myEps = 0.0001
    nParams = len(myParams)
    epsVec = np.zeros(nParams)
    # These are my implemented gradient solutions
    myGrads = cofiGrad(myParams, myY, myR, myNU, myNM, myNF, myLambda)

    # Choose 10 random elements of my combined (X, Theta) param vector
    # and compute the numerical gradient for each... print to screen
    # the numerical gradient next to the my cofiGradient to inspect

    for i in range(10):
        idx = np.random.randint(0, nParams)
        epsVec[idx] = myEps
        loss1 = cofiCostFunc(myParams - epsVec, myY, myR, myNU, myNM, myNF, myLambda)
        loss2 = cofiCostFunc(myParams + epsVec, myY, myR, myNU, myNM, myNF, myLambda)
        myGrad = (loss2 - loss1) / (2*myEps)
        epsVec[idx] = 0
        print('%0.15f \t %0.15f \t %0.15f' % (myGrad, myGrads[idx],myGrad - myGrads[idx]))

# print("Checking gradient with lambda = 0...",checkGradient(flattenParams(X,Theta),Y,R,nu,nm,nf))

# print("Checking gradient with lambda = 0...",checkGradient(flattenParams(X,Theta),Y,R,nu,nm,nf, myLambda=1.5))

# rate some movies
my_ratings = np.zeros((1682,1))
my_ratings[0]   = 4
my_ratings[97]  = 2
my_ratings[6]   = 3
my_ratings[11]  = 5
my_ratings[53]  = 4
my_ratings[63]  = 5
my_ratings[65]  = 3
my_ratings[68]  = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

Y = ex8_movies['Y']
R = ex8_movies['R']
# Use 10 features
nf = 10

# Add ratings to the Y matrix, and the relevant row to the R matrix
myR_row = my_ratings > 0
Y = np.hstack((Y, my_ratings))
R = np.hstack((R, myR_row))
nm, nu = Y.shape

def normalizeRatings(myY, myR):
    """
    Preprocess data by subtracting mean rating for every movie (every row)
    This is important because without this, a user who hasn't rated any movies
    will have a predicted score of 0 for every movie, when in reality
    they should have a predicted score of [average score of that movie].
    """
    # The mean is only counting movies that were rated
    Ymean = np.sum(myY, axis=1)/np.sum(myR, axis=1)
    Ymean = Ymean.reshape((Ymean.shape[0],1))

    return myY - Ymean, Ymean

Ynorm, Ymean = normalizeRatings(Y, R)

# Generate random initial parameters, Theta and X
X = np.random.rand(nm, nf)
Theta = np.random.rand(nu, nf)
myFlat = flattenParams(X, Theta)

# Regularization parameter of 10 is used (as used in the homework assignment)
myLambda = 10.

result = scipy.optimize.fmin_cg(cofiCostFunc, x0=myFlat, fprime=cofiGrad,
                                args=(Y,R,nu,nm,nf,myLambda),
                                maxiter=50, disp=True, full_output=True)

resX, resTheta = reshapeParams(result[0], nm, nu, nf)
# After training the model, now make recommendations by computing
# the predictions matrix
prediction_matrix = resX.dot(resTheta.T)

# Grab the last user's predictions (since I put my predictions at the
# end of the Y matrix, not the front)
# Add back in the mean movie ratings
my_predictions = prediction_matrix[:,-1] + Ymean.flatten()

pred_idxs_sorted = np.argsort(my_predictions)
pred_idxs_sorted[:] = pred_idxs_sorted[::-1]

print('Top recommendations for you: ')
for i in range(20):
    print('Predicting rating %0.1f for movie %s'%(my_predictions[pred_idxs_sorted[i]], movies[pred_idxs_sorted[i]]))

print('\nOriginal ratings provied: ')
for i in range(len(my_ratings)):
    if (my_ratings[i] > 0):
        print('Rated %d for movie %s'%(my_ratings[i], movies[i]))