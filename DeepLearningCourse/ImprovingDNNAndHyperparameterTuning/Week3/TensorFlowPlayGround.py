import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

np.random.seed(1)

y_hat = tf.constant(36, name='y_hat') # Define y_hat constant, set to 36
y = tf.constant(39, name='y') # Define y, set to 39

loss = tf.Variable((y - y_hat)**2, name='loss') # Create a variable for the loss

init = tf.global_variables_initializer()
# When init is run later (session.run(init)),
# the loss variable will be initialized and ready to be computed
# Create a session and print the output
# Initializes the variables
# Prints the loss

# with tf.Session() as session:
#     session.run(init)
#     print(session.run(loss))

# a = tf.constant(2)
# b = tf.constant(10)
# c = tf.multiply(a, b)

# sess = tf.Session()
# print(sess.run(c)

# sess = tf.Session()
# x = tf.placeholder(tf.int64, name='x')
# print(sess.run(2*x, feed_dict={x: 3}))

def linear_function():
    """
    Implements a linear function: 
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns: 
    result -- runs the session for Y = WX + b 
    """

    np.random.seed(1)

    X = tf.constant(np.random.rand(3,1), name='X')
    W = tf.constant(np.random.rand(4,3), name='W')
    b = tf.constant(np.random.rand(4,1), name='b')
    Y = tf.add(tf.matmul(W, X), b)

    sess = tf.Session()
    result = sess.run(Y)

    sess.close()

    return result

# print( "result = " + str(linear_function()))

def sigmoid(z):
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    results -- the sigmoid of z
    """

    x = tf.placeholder(tf.float32, name='x')

    sigmoid = tf.divide(1., tf.add(1., tf.exp(x)))

    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x: -z})

    return result

# print ("sigmoid(0) = " + str(sigmoid(0)))
# print ("sigmoid(12) = " + str(sigmoid(12)))

def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0) 
    
    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels" 
    in the TensorFlow documentation. So logits will feed into z, and labels into y. 
    
    Returns:
    cost -- runs the session of the cost (formula (2))
    """

    # Create the placeholders for "logits" (z) and "labels" (y)
    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    sess = tf.Session()

    result = sess.run(cost, feed_dict={z: logits, y: labels})

    sess.close()

    return result

# logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
# cost = cost(logits, np.array([0,0,1,1]))
# print ("cost = " + str(cost))

def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """

    C = tf.constant(C, name='C')

    one_hot_matrix = tf.one_hot(labels, C, axis=0)

    sess = tf.Session()

    result = sess.run(one_hot_matrix)

    sess.close()

    return result

# labels = np.array([1,2,3,0,2,1])
# one_hot = one_hot_matrix(labels, C = 4)
# print ("one_hot = " + str(one_hot))

def ones(shape):
    """
    Creates an array of ones of dimension shape
    
    Arguments:
    shape -- shape of the array you want to create
        
    Returns: 
    ones -- array containing only ones
    """

    ones = tf.ones(shape)

    sess = tf.Session()

    result = sess.run(ones)

    sess.close()

    return result

print("Ones=", str(ones([3])))