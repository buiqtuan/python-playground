import sys

import numpy as np

# data processing
import pandas as pd

# data visualization
# import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

# algorithm
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

#----------------DATA INFO----------------
"""
The training-set has 891 examples and 11 features + the target variable (survived). 
2 of the features are floats, 5 are integers and 5 are objects. Below I have listed the features with a short description:
survival:   Survival
PassengerId: Unique Id of a passenger.
pclass: Ticket class    
sex:    Sex 
Age:    Age in years    
sibsp:  # of siblings / spouses aboard the Titanic  
parch:  # of parents / children aboard the Titanic  
ticket: Ticket number   
fare:   Passenger fare  
cabin:  Cabin number    
embarked:   Port of Embarkation
"""
testFilePath = 'test.csv'
trainFilePath = 'train.csv'

# check if in debug mode
gettrace = getattr(sys, 'gettrace', None)

if gettrace():
    print('In debug mode!')
    testFilePath = 'D:\workspace\sideprojects\python-playground\Kaggle\RMSTitanic//' + testFilePath 
    trainFilePath = 'D:\workspace\sideprojects\python-playground\Kaggle\RMSTitanic//' + trainFilePath

test_df = pd.read_csv(testFilePath)
train_df = pd.read_csv(trainFilePath)

# print(train_df.info())
# print(train_df.describe())
# print(train_df.head(15))

# Let's take a more detailed look at what data is actually missing:
total = train_df.isnull().sum().sort_values(ascending=False)
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
print(missing_data.head(5))