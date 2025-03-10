import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import *
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def update_parameters_with_gd(parameters, grads, learning_rate):
	"""
	Update parameters using one step of gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- the learning rate, scalar.
    
    Returns:
    parameters -- python dictionary containing your updated parameters
	"""

	L = len(parameters) // 2
	
	for l in range(L):
		parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - learning_rate*grads['dW' + str(l+1)]
		parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - learning_rate*grads['db' + str(l+1)] 

	return parameters

# parameters, grads, learning_rate = update_parameters_with_gd_test_case()

# parameters = update_parameters_with_gd(parameters, grads, learning_rate)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    np.random.seed(seed)
    m = X.shape[1] # Number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # Number of mini batch size
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size : (k+1)*mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, m - (m % mini_batch_size) : ]
        mini_batch_Y = shuffled_Y[:, m - (m % mini_batch_size) : ]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

# X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
# mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)

# print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
# print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
# print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
# print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
# print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 
# print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
# print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))

def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """
    L = len(parameters) // 2 # Number of layers in neural net
    v = {}

    for l in range(L):
        v['dW' + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        v['db' + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)

    return v

# parameters = initialize_velocity_test_case()

# v = initialize_velocity(parameters)
# print("v[\"dW1\"] = " + str(v["dW1"]))
# print("v[\"db1\"] = " + str(v["db1"]))
# print("v[\"dW2\"] = " + str(v["dW2"]))
# print("v[\"db2\"] = " + str(v["db2"]))

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2

    for l in range(L):
        _vdW = v['dW' + str(l+1)]
        _vdW = beta*_vdW + (1 - beta)*grads['dW' + str(l+1)]
        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - learning_rate*_vdW
        v['dW' + str(l+1)] = _vdW

        _vdb = v['db' + str(l+1)]
        _vdb = beta*_vdb + (1 - beta)*grads['db' + str(l+1)]
        parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - learning_rate*_vdb
        v['db' + str(l+1)] = _vdb

    return parameters, v

# parameters, grads, v = update_parameters_with_momentum_test_case()

# parameters, v = update_parameters_with_momentum(parameters, grads, v, beta = 0.9, learning_rate = 0.01)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print("v[\"dW1\"] = " + str(v["dW1"]))
# print("v[\"db1\"] = " + str(v["db1"]))
# print("v[\"dW2\"] = " + str(v["dW2"]))
# print("v[\"db2\"] = " + str(v["db2"]))

def initialize_adam(parameters):
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...
    """

    L = len(parameters) // 2
    v = {}
    s = {}
    
    for l in range(L):
        v['dW' + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        v['db' + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)
        s['dW' + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        s['db' + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)

    return v, s


# parameters = initialize_adam_test_case()

# v, s = initialize_adam(parameters)
# print("v[\"dW1\"] = " + str(v["dW1"]))
# print("v[\"db1\"] = " + str(v["db1"]))
# print("v[\"dW2\"] = " + str(v["dW2"]))
# print("v[\"db2\"] = " + str(v["db2"]))
# print("s[\"dW1\"] = " + str(s["dW1"]))
# print("s[\"db1\"] = " + str(s["db1"]))
# print("s[\"dW2\"] = " + str(s["dW2"]))
# print("s[\"db2\"] = " + str(s["db2"]))

def update_parameters_with_adam(parameters, grads, v, s, t, 
                                learning_rate=.01, beta1=.9, beta2=.999, eps=1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
​
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """

    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v['dW' + str(l+1)] = beta1*v['dW' + str(l+1)] + (1 - beta1)*grads['dW' + str(l+1)]
        v['db' + str(l+1)] = beta1*v['db' + str(l+1)] + (1 - beta1)*grads['db' + str(l+1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected['dW' + str(l+1)] = v['dW' + str(l+1)] / (1 - beta1)**t
        v_corrected['db' + str(l+1)] = v['db' + str(l+1)] / (1 - beta1)**t

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s['dW' + str(l+1)] = beta2*s['dW' + str(l+1)] + (1 - beta2)*grads['dW' + str(l+1)]
        s['db' + str(l+1)] = beta2*s['db' + str(l+1)] + (1 - beta2)*grads['db' + str(l+1)]

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected['dW' + str(l+1)] = s['dW' + str(l+1)] / (1 - beta2)**t
        s_corrected['db' + str(l+1)] = s['db' + str(l+1)] / (1 - beta2)**t

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - learning_rate*(v_corrected['dW' + str(l+1)] / (s_corrected['dW' + str(l+1)] + eps))
        parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - learning_rate*(v_corrected['db' + str(l+1)] / (s_corrected['db' + str(l+1)] + eps))

    return parameters, v, s

# parameters, grads, v, s = update_parameters_with_adam_test_case()
# parameters, v, s  = update_parameters_with_adam(parameters, grads, v, s, t = 2)

# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print("v[\"dW1\"] = " + str(v["dW1"]))
# print("v[\"db1\"] = " + str(v["db1"]))
# print("v[\"dW2\"] = " + str(v["dW2"]))
# print("v[\"db2\"] = " + str(v["db2"]))
# print("s[\"dW1\"] = " + str(s["dW1"]))
# print("s[\"db1\"] = " + str(s["db1"]))
# print("s[\"dW2\"] = " + str(s["dW2"]))
# print("s[\"db2\"] = " + str(s["db2"]))

train_X, train_Y = load_dataset()

def model(X, Y, layers_dims, optimizer, learning_rate=.0007, mini_batch_size=64,
          beta=.9, beta1=.9, beta2=.999, eps=1e-8, num_epochs=10000, print_cost=True):
    """
    3-layer neural network model which can be run in different optimizer modes.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs
​
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """

    L = len(layers_dims)
    costs = []
    t = 0
    seed = 10

    # Initialize parameters
    parameters = initialize_parameters(layers_dims)
    # Initialize the optimizer
    if optimizer == 'gd':
        pass # No intialization required
    elif optimizer == 'momentum':
        v = initialize_velocity(parameters)
    elif optimizer == 'adam':
        v, s = initialize_adam(parameters)
    # Optimization loop
    for i in range(num_epochs):
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        for mini_batch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = mini_batch
            # Forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)
            # Compute cost
            cost = compute_cost(a3, minibatch_Y)
            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)
            # Update parameters
            if optimizer == 'gd':
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == 'momentum':
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == 'adam':
                t = t + 1
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, eps)

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters

# Mini-batch Gradient descent
"""
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "gd")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
"""

# Mini-batch gradient descent with momentum
"""
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
"""

# Mini-batch with Adam mode

# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)