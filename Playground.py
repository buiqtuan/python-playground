import numpy as np
import itertools

# x = [1,2,3,4,5,6,7,8,9]



# tinyTuple = ('a',1,1,'b')

# tinydict = {'name': 'john','code':6734, 'dept': 'sales'}


# for ind, val in enumerate(x,start=0) :
#     print("{} and {}".format(ind, val))

# for ind,val in enumerate(range(1,8,2)) :
#     print("{} and {}".format(ind, val))

# w, h = 8, 5
# Matrix = [[x for x in range(0,w,1 )] for y in range(h)]

# print(len(range(1,8,2)))

# dictionary = { "some_key": "some_value" }

# print(dictionary.items())

# for key in dictionary:
#     print("%s --> %s" %(key, dictionary[key]))

# for x in range(1,8,3) :
# 	print(x)

# for ind,val in enumerate(tinyTuple,start = 0) :
# 	print('{} and {}'.format(ind,val))

# def printingLocalTime() :
# 	print(time.localtime(time.time())[0])

# print(int('123123123111111111') + 1)

# test decorator
# def decorator_func(say_hello_func):
# 	def wrapper_func(hello_war, world_war):
# 		hello = 'hello'
# 		world = 'world'

# 		if not hello_war:
# 			hello_war = hello

# 		if not world_war:
# 			world_war = world
		
# 		return say_hello_func(hello_war, world_war)
	
# 	return wrapper_func

# @decorator_func
# def say_hello(hello_war, world_war):
# 	print(hello_war + " " + world_war)

# say_hello(None, None)

# y = [0,2,1,4,6,8,4,-1]
# x = np.array([[2],[1],[3]])

# print(y[1:])

# print(y[:3])

# print(y[:-1])

# print(y[-1:])

# print(x[:,0])

# x = np.linspace(-1,1.5,50)

# print(np.meshgrid(x))

# A = np.array([[1,2,3],[4,5,6]])
# B = np.zeros(A.shape)

# B[0:,0:] = A

# print(A)
# print(B)
# None, False, '' and 0 equal to False

# a = [0]*2

# print(a)

# assert 1 == 2

# a = np.array([[1], [3], [5,6]])

# print(list(itertools.chain.from_iterable(a.flatten())))

# a = np.array([1,2])

# b = np.array([1,2])

# c = np.array([[1],[2],[3]])

# d = np.insert(c,c.shape[1],-1, axis=1)
# print(c.shape)
# print(d.shape)
# print(c)
# print(d[:,1])

# print(np.std([1,3,4,6], ddof=1))

# a = 'a                b  c'

# print(a.split())

# a = [1,2,3,4,5,1,1]

# b = np.array([1,2,3,4,5,1,1])

# print(sum(b==1))

# print(sum([i for i in a if i == 1]))

a = np.array([0,1,2,3,4])

print(a[::-1])