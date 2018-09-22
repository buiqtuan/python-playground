from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

def loadData():
    dataFile = open('D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssigment2\ex2data1.txt','r')

    dataArray = dataFile.read().splitlines()

    dataFile.close()

    trainingSet = [[1 for k in range(len(dataArray))], 
                [(float(i.split(',')[0])) for i in dataArray],
                [(float(j.split(',')[1])) for j in dataArray]]

    resultSet = [int(data.split(',')[2]) for data in dataArray]

    return (trainingSet, resultSet)

def plotData(_T):
    figure = plt.figure()
    ax = figure.add_subplot(111)
    temp1, temp2 = [], []

    for ind,val in enumerate(resultSet):
        if (val == 1):
            temp1.append([_T[1][ind],[_T[2][ind]]])
        else:
            temp2.append([_T[1][ind],[_T[2][ind]]])

    ax.scatter([k[0] for k in temp1],[k[1] for k in temp1], marker='s', color='#2e91be')
    ax.scatter([k[0] for k in temp2],[k[1] for k in temp2], marker='d', color='#d46f9f')

    plt.axis([30, 105, 30, 105])
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def costFucntion(_T, _R, alpha):
    m = len(_R)
    theta = [0 for i in range(3)]
    tempTheta = [0 for i in range(3)]
    cost = 0
    iterations = 1500

    #calculate theta:
    while (iterations > 0):

        for i in range(3):
            #sigma (X - Y)
            sigmaXY = 0
            for j in range(m):
                _thetaT = np.array(theta)
                _X = np.array([[_T[0][j], _T[1][j], _T[2][j]]]).T
                z = np.dot(_thetaT,_X)[0]
                sigmaXY = sigmaXY + (sigmoid(z) - _R[j])*_T[i][j]

            tempTheta[i] = theta[i] - (alpha / m)*sigmaXY

        for i in range(m):
            _thetaT = np.array(theta)
            _X = np.array([[_T[0][j], _T[1][j], _T[2][j]]]).T
            z = np.dot(_thetaT,_X)[0]
            cost = cost + (_R[i]*np.log(sigmoid(z))) + (1 - _R[i])*np.log(1 - sigmoid(z))

        theta = tempTheta
        iterations = iterations - 1 

    print(cost / m)
    return (theta, cost/m)

T = loadData()[0]
R = loadData()[1]
print(costFucntion(T,R,.01))