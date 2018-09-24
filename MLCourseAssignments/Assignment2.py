from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pandas
from scipy import optimize

datafile = './DataAssigment2/ex2data1.txt'
#!head $datafile
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True) #Read in comma separated data
#Form the usual "X" matrix and "y" vector
X = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
m = y.size # number of training examples
#Insert the usual column of 1's into the "X" matrix
X = np.insert(X,0,1,axis=1)

#Divide the sample into two: ones with positive classification, one with null classification
pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])

inital_theta = np.zeros((X.shape[1],1))

def plotData():
    plt.figure(figsize=(10,6))
    plt.plot(pos[:,1],pos[:,2],'k+',label='Admitted')
    plt.plot(neg[:,1],neg[:,2],'yo',label='Not admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    plt.grid(True)

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def h(myT, myX):
    return sigmoid(np.dot(myX, myT))

def computeCost(myT, myX, myY, myLambda = 0):
    #note to self: *.shape is (rows, columns)
    term1 = np.dot(-np.array(myY).T,np.log(h(myT,myX)))
    term2 = np.dot((1-np.array(myY)).T,np.log(1-h(myT,myX)))
    regterm = (myLambda/2) * np.sum(np.dot(myT[1:].T,myT[1:])) #Skip theta0
    return float( (1./m) * ( np.sum(term1 - term2) + regterm ) )

def optimizeTheta(mytheta,myX,myy,mylambda=0.):
    result = optimize.fmin(computeCost, x0=mytheta, args=(myX, myy, mylambda), maxiter=400, full_output=True)
    return (result[0], result[1])

# theta, mincost = optimizeTheta(inital_theta, X, y)

# Visualize decision boundary

# boundary_xs = np.array([np.min(X[:,1]), np.max(X[:,1])])
# boundary_ys = (-1./theta[2])*(theta[0] + theta[1]*boundary_xs)
# print(boundary_xs, boundary_ys)
# plotData()
# plt.plot(boundary_xs,boundary_ys,'b-',label='Decision Boundary')
# plt.legend()
# plt.show()

# def loadData():
#     dataFile = open('D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssigment2\ex2data1.txt','r')

#     dataArray = dataFile.read().splitlines()

#     dataFile.close()

#     trainingSet = [[1 for k in range(len(dataArray))], 
#                 [(float(i.split(',')[0])) for i in dataArray],
#                 [(float(j.split(',')[1])) for j in dataArray]]

#     resultSet = [int(data.split(',')[2]) for data in dataArray]

#     return (trainingSet, resultSet)

# def plotData(_T):
#     figure = plt.figure()
#     ax = figure.add_subplot(111)
#     temp1, temp2 = [], []

#     for ind,val in enumerate(resultSet):
#         if (val == 1):
#             temp1.append([_T[1][ind],[_T[2][ind]]])
#         else:
#             temp2.append([_T[1][ind],[_T[2][ind]]])

#     ax.scatter([k[0] for k in temp1],[k[1] for k in temp1], marker='s', color='#2e91be')
#     ax.scatter([k[0] for k in temp2],[k[1] for k in temp2], marker='d', color='#d46f9f')

#     plt.axis([30, 105, 30, 105])
#     plt.xlabel('Exam 1 score')
#     plt.ylabel('Exam 2 score')
#     plt.show()

# using with customed learning rate
# def costFucntion(_T, _R, alpha):
#     m = len(_R)
#     theta = [0 for i in range(3)]
#     cost = 0
#     _cost = 0
#     iterations = 400

#     #calculate theta:
#     while (iterations > 0):
#         tempTheta = [0 for i in range(3)]
#         for i in range(3):
#             #sigma (X - Y)
#             sigmaXY = 0
#             cost = 0
#             for j in range(m):
#                 _thetaT = np.array(theta)
#                 _X = np.array([[_T[0][j], _T[1][j], _T[2][j]]]).T
#                 z = np.dot(_thetaT,_X)[0]
#                 sigmoidValue = sigmoid(z)
#                 sigmaXY = sigmaXY + (sigmoidValue - _R[j])*_T[i][j]

#                 if (i == 0):
#                     if (_R[j] == 1):   
#                         cost = cost + (_R[j]*np.log(sigmoidValue))
#                     else:
#                         cost = cost + (1 - _R[j])*np.log(1 - sigmoidValue)

#             tempTheta[i] = theta[i] - (alpha / m)*sigmaXY

#         theta = tempTheta
#         iterations = iterations - 1 

#     return (theta, - cost/m)