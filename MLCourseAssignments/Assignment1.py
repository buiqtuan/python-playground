import numpy
import matplotlib.pyplot as plt

# create identity matrix

# identityMatrix = numpy.identity(5)

# print(identityMatrix)

def calculateTheta1():
	dataFile = open('D:\workspace\sideprojects\python-playground\MLCourseAssignments\Assignment1Data.txt','r')

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
	dataFile = open('D:\workspace\sideprojects\python-playground\MLCourseAssignments\data\Assignment2Data.txt','r')

	dataArray = dataFile.read().splitlines()

	sizeArray = [float(data.split(',')[0]) for data in dataArray]

	m = len(sizeArray)
	# nomarlizing
	avarageInput = sum(sizeArray)/m
	rangeInput = max(sizeArray) - min(sizeArray)

	dataFile.close()

	trainingSet = [[1 for k in range(len(dataArray))],
				  [((float(data.split(',')[0]) - avarageInput)/rangeInput) for data in dataArray], 
				  [float(data.split(',')[1]) for data in dataArray]] #size and no of bedrooms
	outcomeSet = [float(data.split(',')[2]) for data in dataArray] #prices
	
	theta = [0,0,0]
	costValue = 0
	iterations = 2000
	alpha = .01

	while (iterations > 0):
		temp = [0,0,0]
		sumTheta = [0,0,0]

		tempArray = calculateUnitTheta(theta, trainingSet)

		for i in range(len(sumTheta)):
			for j in range(m):
				sumTheta[i] = sumTheta[i] + (tempArray[0][j] - outcomeSet[j])*trainingSet[i][j]

			temp[i] = theta[i] - (alpha/m)*sumTheta[i]

		theta = temp

		for i in range(m):
			costValue = costValue + (theta[0]*trainingSet[0][i] + theta[1]*trainingSet[1][i] + theta[2]*trainingSet[2][i] - outcomeSet[i])**2

		iterations = iterations - 1

	return (theta, costValue/(2*m))
		
def calculateUnitTheta(_theta,_X):
	return numpy.dot([_theta], _X)
	

print(calculateTheta2())

# print(trainingSet)

# plt.plot(trainingSet[1], outcomeSet[0], 'ro')
# plt.axis([0, 25, -20, 20])
# plt.ylabel('Revenues')
# plt.xlabel('Populations')
# plt.show()