import numpy
import matplotlib.pyplot as plt

# create identity matrix

# identityMatrix = numpy.identity(5)

# print(identityMatrix)

dataFile = open('./data/Assignment1Data.txt','r')

dataArray = dataFile.read().splitlines()

dataFile.close()

trainingSet = numpy.matrix([[1 for k in range(len(dataArray))],[float(data.split(',')[0]) for data in dataArray]]) #populations
outcomeSet = numpy.matrix([float(data.split(',')[1]) for data in dataArray]) #revenues

iterations = 1500
alpha = .01

print(trainingSet)

plt.plot(trainingSet[1], outcomeSet[0], 'ro')
plt.axis([0, 25, -20, 20])
plt.ylabel('Revenues')
plt.xlabel('Populations')
plt.show()