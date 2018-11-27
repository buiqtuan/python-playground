import numpy as np
from gc_utils import *

def forward_propagation(x, theta):
	"""
	Implement the linear forward propagation (compute J) presented in Figure 1 (J(theta) = theta * x)
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    
    Returns:
    J -- the value of function J, computed using the formula J(theta) = theta * x
	"""

	J = np.dot(theta,x)

	return J

# x, theta = 2, 4
# J = forward_propagation(x, theta)
# print ("J = " + str(J))

def backward_propagation(x, theta):
	"""
	Computes the derivative of J with respect to theta (see Figure 1).
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    
    Returns:
    dtheta -- the gradient of the cost with respect to theta
	"""

	dtheta = x

	return dtheta

# x, theta = 2, 4
# dtheta = backward_propagation(x, theta)
# print ("dtheta = " + str(dtheta))

