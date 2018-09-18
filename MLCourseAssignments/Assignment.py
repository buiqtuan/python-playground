import numpy
import matplotlib.pyplot as plt

# create identity matrix

# identityMatrix = numpy.identity(5)

# print(identityMatrix)

def calculateTheta1():
	dataFile = open('./data/Assignment1Data.txt','r')

	dataArray = dataFile.read().splitlines()

	dataFile.close()

	trainingSet = [[1 for k in range(len(dataArray))],[float(data.split(',')[0]) for data in dataArray]] #populations
	outcomeSet = [float(data.split(',')[1]) for data in dataArray] #revenues

	iterations = 1500
	alpha = .01
	thetaZero = thetaOne = 0
	m = len(outcomeSet)
	costValue = 0
	
	while (iterations > 0):
		sumThetaZero = sumThetaOne = costValue = 0
		for i in range(m):
			sumThetaZero = sumThetaZero + thetaZero + thetaOne*trainingSet[1][i] - outcomeSet[i]
			sumThetaOne = sumThetaOne + (thetaZero + thetaOne*trainingSet[1][i] - outcomeSet[i])*trainingSet[1][i]
		
		temp0 = thetaZero - (alpha/m)*sumThetaZero
		temp1 = thetaOne - (alpha/m)*sumThetaOne

		for i in range(m):
			costValue = costValue + (thetaZero + thetaOne*trainingSet[1][i] - outcomeSet[i])**2

		if (costValue/(2*m) <= .001):
			break

		thetaZero = temp0
		thetaOne = temp1
		iterations = iterations - 1

	return (thetaZero, thetaOne, costValue/(2*m))
		

def calculateTheta2():
	dataFile = open('./data/Assignment2Data.txt','r')

	dataArray = dataFile.read().splitlines()

	dataFile.close()

	

# print(trainingSet)

# plt.plot(trainingSet[1], outcomeSet[0], 'ro')
# plt.axis([0, 25, -20, 20])
# plt.ylabel('Revenues')
# plt.xlabel('Populations')
# plt.show()