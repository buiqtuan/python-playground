import numpy as np

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

x = np.linspace(-1,1.5,50)

print(np.meshgrid(x))