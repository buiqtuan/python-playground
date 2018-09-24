from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

import pickle

dataPath = '../TrainingData/BirdImages'
trainingImgDict = {}

with open(dataPath + '/data_batch_1','rb') as f:
	trainingImgDict.update(pickle.load(f,encoding = 'bytes'))

print(len(trainingImgDict))