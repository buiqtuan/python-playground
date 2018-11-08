import sys
import re

import numpy as np
np.random.seed(7)

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

#----------------- ANALYZING DATA -----------------
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
data = [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)

# print(train_df['not_alone'].value_counts())

# axes = sns.factorplot('relatives', 'Survived', data=train_df, aspect=2.5)

#----------------- DATA PREPROCESSING -----------------
# PassengerId

train_df = train_df.drop(['PassengerId'], axis=1)

# CABIN
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)

# Drop the cabin feature
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)

# Age
"""
Now we can tackle the issue with the age features missing values.
I will create an array that contains random numbers, which are computed based on the mean age value in regards to the standard deviation and is_null.
"""
data = [train_df, test_df]

for dataset in data:
    mean = train_df['Age'].mean()
    std = test_df['Age'].std()
    is_null = dataset['Age'].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size=is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset['Age'].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset['Age'] = age_slice
    dataset['Age'] = train_df['Age'].astype(int)

# Embarked
"""
Since the Embarked feature has only 2 missing values, we will just fill these with the most common one.
"""
# print(train_df['Embarked'].describe())
common_value = 'S'
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

#----------------- CONVERTING FEATURES -----------------
# print(train_df.info())
"""
Above you can see that 'Fare' is a float and we have to deal with 4 categorical features: Name, Sex, Ticket and Embarked.
Lets investigate and transfrom one after another.
"""
# Fare:
# Converting "Fare" from float to int64, using the "astype()" function pandas provides:

data = [train_df, test_df]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)

# Name:
# We will use the Name feature to extract the Titles from the Name, so that we can build a new feature out of that.
data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)

# Sex
# Convert 'Sex' feature into numeric.
genders = {"male": 0, "female": 1}
data = [train_df, test_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)

# Ticket
# print(train_df['Ticket'].describe())
"""
Since the Ticket attribute has 681 unique tickets, 
it will be a bit tricky to convert them into useful categories. 
So we will drop it from the dataset.
"""
train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)

# Embarked:
# Convert 'Embarked' feature into numeric.

ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)

#----------------- CREATING CATEGORIES -----------------

# Age
"""
Now we need to convert the 'age' feature. First we will convert it from float into integer.
Then we will create the new 'AgeGroup" variable, by categorizing every age into a group.
Note that it is important to place attention on how you form these groups,
since you don't want for example that 80% of your data falls into group 1.
"""
data = [train_df, test_df]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

# print(train_df['Age'].value_counts())

# Fare:
"""
For the 'Fare' feature, we need to do the same as with the 'Age' feature.
But it isn't that easy, because if we cut the range of the fare values into a few equally big categories,
80% of the values would fall into the first category.
Fortunately, we can use sklearn "qcut()" function, that we can use to see, how we can form the categories.
"""
data = [train_df, test_df]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

# print(train_df.head(10))

#----------------- CREATING NEW FEATURES -----------------

# Age * Class
data = [train_df, test_df]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']
# Fare per Person
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)

# print(train_df.head(20))

#----------------- BUILDING MACHINE LEARNING MODELS -----------------

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

# Stochastic gradient descent (SGD) learning
sgd = linear_model.SGDClassifier(max_iter=50, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100 ,2)

print(round(acc_sgd, 2), '%')