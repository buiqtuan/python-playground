from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

import pickle

# dataPath = '../TrainingData/BirdImages'
dataPath = 'D:\workspace\sideprojects\python-playground\TrainingData\BirdImages'

# trainingImgDict = {}
# for i in range(4):
# 	with open(dataPath + '\data_batch_{}'.format(i+1),'rb') as f:
# 		trainingImgDict.update(pickle.load(f,encoding = 'bytes'))

X, Y, X_test, Y_test = pickle.load(open(dataPath + '\\full_dataset.pkl','rb'), encoding='bytes')

X,Y = shuffle(X, Y)

# normalizing data
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep, data_augmentation=img_aug)

# First convolution
network = conv_2d(network, 32, 3, activation='relu')

# First Max pooling
network = max_pool_2d(network, 2)

# Second convolution
network = conv_2d(network, 64, 3, activation='relu')

# Third convolution
network = conv_2d(network, 64, 3, activation='relu')

# Second Max pooling
network = max_pool_2d(network, 2)

# Generate fully connected 512 node neural network
network = fully_connected(network, 512, activation='relu')

# Drop out - throw away some data randomly during training to prevent over-fitting
network = dropout(network, .5)

# Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
network = fully_connected(network, 2, activation='softmax')

# Set tflearn to use regression
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=.001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='BirdDetector.tfl.ckpt')

# Train. With 100 training passes
model.fit(X, Y, 
			n_epoch=100, 
			shuffle=True, 
			validation_set=(X_test, Y_test), 
			show_metric=True, 
			batch_size=96, 
			snapshot_epoch=True, run_id='BirdDetector')

# Save model when training is complete to a file
model.save("BirdDetector.tfl")
print("Network trained and saved as BirdDetector.tfl!")