from __future__ import print_function

import numpy as np 
from time import time

d, N = 1000, 10000
X = np.random.rand(N,d)
z = np.random.rand(d)

# naively compute square distance between two vector
def dist_pp(z,x):
	d = z - x.reshape(z.shape) # force x and z to have the same dims
	return np.sum(d*d)

# from one point to each point in a set, naive
def dist_ps_naive(z, X):
	N = X.shape[0]
	res = np.zeros((1, N))
	for i in range(N):
		res[0][i] = dist_pp(z, X[i])
	return res

# from one point to each point in a set, fast
def dist_ps_fast(z, X):
	X2 = np.sum(X*X, 1) # square of l2 norm of each ROW of X
	z2 = np.sum(z*z) # square of l2 norm of z
	return X2 + z2 - 2*X.dot(z) # z2 can be ignored

# t1 = time()
# D1 = dist_ps_naive(z, X)
# print('naive point2set, running time:', time() - t1, 's')
# t1 = time()
# D2 = dist_ps_fast(z, X)
# print('fast point2set , running time:', time() - t1, 's')
# print('Result difference:', np.linalg.norm(D1 - D2))

M = 100
Z = np.random.randn(M, d)

# from each point in one set to each point in another set, half fast
def dist_ss_0(Z, X):
	M = Z.shape[0]
	N = X.shape[0]
	res = np.zeros((M, N))
	for i in range(M):
		res[i] = dist_ps_fast(Z[i], X)
	return res

# from each point in one set to each point in another set, fast
def dist_ss_fast(Z, X):
	X2 = np.sum(X*X, 1) # square of l2 norm of each ROW of X
	Z2 = np.sum(Z*Z, 1) # square of l2 norm of each ROW of Z
	return Z2.reshape(-1, 1) + X2.reshape(1, -1) - 2*Z.dot(X.T)

t1 = time()
D3 = dist_ss_0(Z, X)
print('half fast set2set running time:', time() - t1, 's')
t1 = time()
D4 = dist_ss_fast(Z, X)
print('fast set2set running time', time() - t1, 's')
print('Result difference:', np.linalg.norm(D3 - D4))