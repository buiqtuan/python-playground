from __future__ import print_function

from PIL import Image
import sys
import random
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import io as io
from scipy import optimize
from sklearn import svm
import time
import re
import nltk


spamTestFilePath = './DataAssignment6/spamTest.mat'
spamTrainFilePath = './DataAssignment6/spamTrain.mat'
vocabDictPath = './DataAssignment6/vocab.txt'

# Check if in debug mode
gettrace = getattr(sys, 'gettrace', None)

if gettrace():
    print('In Debug Mode!')
    spamTestFilePath = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment6\spamTest.mat'
    spamTrainFilePath = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment6\spamTrain.mat'
    vocabDictPath = 'D:\workspace\sideprojects\python-playground\MLCourseAssignments\DataAssignment6\vocab.txt'

# check that nltk modules has been downloaded?
try:
	nltk.data.find('tokenizers/punkt')
except LookupError:
	print('nltk models has not been downloaded!! -> process downloading models')
	nltk.download()

sampleEmail1 = open('./DataAssignment6/emailSample1.txt','r')

# print(sampleEmail1.read())

def preProcess(email: str):
    """
    Function to do some pre processing (simplification of e-mails).
    Comments throughout implementation describe what it does.
    Input = raw e-mail
    Output = processed (simplified) email
    """
    # Make entire email to lower case
    email = email.lower()
    
    # Strip html tags (strings that look like <blah> where 'blah' does not
    # contain '<' or '>')... replace with a space
    email = re.sub('<[^<>]+>', ' ', email)

    # Replace any number with a string 'number'
    email = re.sub('[0-9]+', 'number', email)

    # Anything starting with http or https:// replaced with 'httpaddr'
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)

    # Strings with "@" in the middle are considered emails --> 'emailaddr'
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)

    # Replace $ with 'dollar'
    email = re.sub('[$]+' , 'dollar', email)

    return email

def emailToTokenList(email: str):
    """
    Function that takes in preprocessed (simplified) email, tokenizes it,
    stems each word, and returns an (ordered) list of tokens in the e-mail
    """

    # Use the NLTK stemmer because it more accurately duplicates the
    # performance of the OCTAVE implementation in the assignment
    stemmer = nltk.stem.porter.PorterStemmer()

    email = preProcess(email)

    # Split the e-mail into individual words (tokens) (split by the delimiter ' ')
    # but also split by delimiters '@', '$', '/', etc etc
    # Splitting by many delimiters is easiest with re.split()
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)

    # Loop over each word (token) and use a stemmer to shorten it,
    # then check if the word is in the vocab_list... if it is,
    # store what index in the vocab_list the word is
    tokenList = []
    for token in tokens:
        # Remove any non alphanumeric characters
        token = re.sub('[^a-zA-Z0-9]', '', token)

        # Use the Porter stemmer to stem the word
        stemmed = stemmer.stem(token)

        # Throw out empty tokens
        if not len(token):
            continue
        
        # Store a list of all unique stemmed words
        tokenList.append(stemmed)

    return tokenList

# print(emailToTokenList(sampleEmail1.read()))

def getVocabDict(reversed=False):
    """
    Function to read in the supplied vocab list text file into a dictionary.
    I'll use this for now, but since I'm using a slightly different stemmer,
    I'd like to generate this list myself from some sort of data set...
    Dictionary key is the stemmed word, value is the index in the text file
    If "reverse", the keys and values are switched.
    """

    vocab_dict = {}
    with open(vocabDictPath, 'r') as f:
        for line in f:
            (val, key) = line.split()
            if not reversed:
                vocab_dict[key] = int(val)
            else:
                vocab_dict[int(val)] = key

    return vocab_dict

def emailToVocabIndices(email, vocab_list):
    """
    Function that takes in a raw email and returns a list of indices corresponding
    to the location in vocab_dict for each stemmed word in the email.
    """
    tokenList = emailToTokenList(email)
    indexList = [vocab_list[token] for token in tokenList if token in vocab_list]
    return indexList


print(emailToVocabIndices(sampleEmail1.read(), getVocabDict()))