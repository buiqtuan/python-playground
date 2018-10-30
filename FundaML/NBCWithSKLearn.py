from __future__ import print_function
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import numpy as np 

# MultinomialNB: count how many times that words appear
# BernoulliNB: only cares that word appears or not

# train data for MultinomialNB
# d1 = [2, 1, 1, 0, 0, 0, 0, 0, 0]
# d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
# d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
# d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]

# train data for BernoulliNB
d1 = [1, 1, 1, 0, 0, 0, 0, 0, 0]
d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]
train_data = np.array([d1, d2, d3, d4])
label = np.array(['B', 'B', 'B', 'N'])

# test data for MultinomialNB
# d5 = np.array([[2, 0, 0, 1, 0, 0, 0, 1, 0]])
# d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])

# test data for BernoulliNB
d5 = np.array([[1, 0, 0, 1, 0, 0, 0, 1, 0]])
d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])

# model = MultinomialNB()

model = BernoulliNB()

model.fit(train_data, label)

print('Predicting class of d5:', str(model.predict(d5)[0]))
print('Probability of d6 in each class:', model.predict_proba(d6))