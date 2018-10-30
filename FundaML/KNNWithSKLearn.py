from __future__ import print_function
import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split # for splitting data
from sklearn.metrics import accuracy_score # for evaluating results

np.random.seed(7)
iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target
print('Labels: ', np.unique(iris_Y))

# split train and test
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_Y, test_size=130)
print('Train size:', X_train.shape[0], ', test size:', X_test.shape[0])

# 1NN with n_neighbors=1, p=2 is l2 norm
# model = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# print("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

def myWeight(distance):
	sigma2 = .4
	return np.exp(-distance**2/sigma2)

# 7NN with n_neighbors=7, p=2 is l2 norm
# if 'weights' is not defined, every data point has the same weight (weights = 'uniform')
# weights = distance is every point has a weights based on its distance to the input data
# we can self-define a way to compute weight, ex: myWeight function above
model = neighbors.KNeighborsClassifier(n_neighbors=7, p=2, weights=myWeight)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy of 7NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))