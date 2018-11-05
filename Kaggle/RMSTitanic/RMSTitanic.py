import sys

import numpy as np

# data processing
import pandas as pd

# data visualization
import seaborn as sns
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

#----------------- VISUALIZING DATA -----------------
# print(train_df.info())
# print(train_df.describe())
# print(train_df.head(15))

# Let's take a more detailed look at what data is actually missing:
# total = train_df.isnull().sum().sort_values(ascending=False)
# percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
# percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
# missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
# print out the most 5 features have the highest rates of missing values
# print(missing_data.head(5))

# print(train_df.columns.values)

#----------------- ANALyZING DATA -----------------
# 1. Sex and Age
# survived = 'survived'
# not_survived = 'not_survived'
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
# women = train_df[train_df['Sex'] == 'female']
# men = train_df[train_df['Sex'] == 'male']

# ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(),
#                 bins=18, label=survived, ax=axes[0], kde=False)
# ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(),
#                 bins=40, label=not_survived, ax=axes[0], kde=False)
# ax.legend()
# ax.set_title('Female')

# ax = sns.distplot(men[men['Survived'] == 1].Age.dropna(),
#                 bins=18, label=survived, ax=axes[1], kde=False)
# ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(),
#                 bins=40, label=not_survived, ax=axes[1], kde=False)
# ax.legend()
# ax.set_title('Male')

# plt.show()

# 2. Embarked, Pclass and Sex:
# FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=4.5, aspect=1.6)
# FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None, order=None, hue_order=None)
# FacetGrid.add_legend()

# Pclass 
# sns.barplot(x='Pclass', y='Survived', data=train_df)

# Relation between Age and Pclass
# grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()

# 3. SibSp and Parch
_data = [train_df, test_df]
for dataset in _data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)

print(train_df['not_alone'].value_counts())

axes = sns.factorplot('relatives', 'Survived', data=train_df, aspect=2.5)
plt.show()