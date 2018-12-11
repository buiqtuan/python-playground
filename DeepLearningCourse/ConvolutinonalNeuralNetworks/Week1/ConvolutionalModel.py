import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values=0)

    return X_pad

# np.random.seed(1)
# x = np.random.randn(4, 3, 3, 2)
# x_pad = zero_pad(x, 2)
# print ("x.shape =", x.shape)
# print ("x_pad.shape =", x_pad.shape)
# print ("x[1,1] =", x[1,1])
# print ("x_pad[1,1] =", x_pad[1,1])

# fig, axarr = plt.subplots(1, 2)
# axarr[0].set_title('x')
# axarr[0].imshow(x[0,:,:,0])
# axarr[1].set_title('x_pad')
# axarr[1].imshow(x_pad[0,:,:,0])

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """

    s = np.multiply(a_slice_prev, W)

    Z = np.sum(s)

    Z = Z + float(b)

    return Z

# np.random.seed(1)
# a_slice_prev = np.random.randn(4, 4, 3)
# W = np.random.randn(4, 4, 3)
# b = np.random.randn(1, 1, 1)

# Z = conv_single_step(a_slice_prev, W, b)
# print("Z =", Z)

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from 'hparameters'
    stride = hparameters['stride']
    pad = hparameters['pad']

    # Compute the dimensions of the CONV output volume using the formula given above
    n_H = int((n_H_prev - f + 2*pad) / stride) + 1
    n_W = int((n_W_prev - f + 2*pad) / stride) + 1

    # Initialize the output volume Z with zeros.
    Z = np.zeros((m, n_H, n_W, n_C))

    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)

    # loop over the batch of training examples
    for i in range(m):
        # Select ith training example's padded activation
        a_prev_pad = A_prev_pad[i]
        # loop over vertical axis of the output volume
        for h in range(n_H):
            # loop over horizontal axis of the output volume
            for w in range(n_W):
                # loop over channels (= #filters) of the output volume
                for c in range(n_C):
                    vert_start = h 
                    vert_end = h + f
                    horiz_start = w
                    horiz_end = w + f

                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell).
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[...,c], b[...,c])

    # Making sure your output shape is correct
    assert(Z.shape == (m, n_H, n_W, n_C))

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache

np.random.seed(1)
A_prev = np.random.randn(10,4,4,3)
W = np.random.randn(2,2,3,8)
b = np.random.randn(1,1,1,8)
hparameters = {"pad" : 2, "stride": 2}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
# print("Z's mean =", np.mean(Z))
# print("Z[3,2,1] =", Z[3,2,1])
# print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])

def pool_forward(A_prev, hparameters, mode='max'):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """

    # Retrieve demensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve hyperparameters from 'hparameters'
    f = hparameters['f']
    stride = hparameters['stride']
    
    # Define the demensions of the output
    n_H = int(1 + (n_H_prev - f)/stride)
    n_W = int(1 + (n_W_prev - f)/stride)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h
                    vert_end = h + f
                    horiz_start = w
                    horiz_end = w + f

                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == 'average':
                        A[i, h, w, c] = np.mean(a_prev_slice)

    # Store the input and hparameters in 'cache' for pool_backward()
    cache = (A_prev, hparameters)

    # Making sure the output shape is correct
    assert(A.shape == (m, n_H, n_W, n_C))

    return A, cache

# np.random.seed(1)
# A_prev = np.random.randn(2, 4, 4, 3)
# hparameters = {"stride" : 2, "f": 3}

# A, cache = pool_forward(A_prev, hparameters)
# print("mode = max")
# print("A =", A)
# print()
# A, cache = pool_forward(A_prev, hparameters, mode = "average")
# print("mode = average")
# print("A =", A)

def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """
    # Retrieve information from 'cache'
    (A_prev, W, b, hparameters) = cache

    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from 'hparameters'
    stride = hparameters['stride']
    pad = hparameters['pad']

    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape

    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h
                    vert_end = h + f
                    horiz_start = w
                    horiz_end = w + f

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:]

                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:] += W[...,c]*dZ[i, h, w, c]
                    dW[...,c] += W[...,c]
                    db[...,c] += b[...,c]

        # Set the ith training example's dA_prev to the unpaded da_prev
        dA_prev[i,:,:,:] = da_prev_pad[pad:-pad, pad:-pad, :]

    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db

np.random.seed(1)
dA, dW, db = conv_backward(Z, cache_conv)
print("dA_mean =", np.mean(dA))
print("dW_mean =", np.mean(dW))
print("db_mean =", np.mean(db))

def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    
    Arguments:
    x -- Array of shape (f, f)
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    k = np.max(x)

    mask = np.array([[0 if (i < k) else 1 for i in j] for j in x])

    return mask

# np.random.seed(1)
# x = np.random.randn(2,3)
# mask = create_mask_from_window(x)
# print('x = ', x)
# print("mask = ", mask)

def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape
    
    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    
    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """

    (n_H, n_W) = shape

    average = float(dz/(n_H*n_W))

    a = np.array([[average for i in range(n_H)] for j in range(n_H)])

    return a

# a = distribute_value(2, (2,2))
# print('distributed value =', a)

def pool_backward(dA, cache, mode='max'):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """

    # Retrieve information from cache
    (A_prev, hparameters) = cache

    # Retriev hyperparameters from 'hparamters'
    stride = hparameters['stride']
    f = hparameters['f']

    # Retrieve demensions from A_prev's shape and dA's shape
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    # Initialize dA_prev with zeros
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h
                    vert_end = h + f
                    horiz_start = w
                    horiz_end = w + f

                    if mode == 'max':
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end,c]

                        mask = create_mask_from_window(a_prev_slice)

                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(dA[i, vert_start: vert_end, horiz_start: horiz_end, c], mask)
                    elif mode == 'average':
                        da = np.sum(dA[i, vert_start:vert_end, horiz_start:horiz_end, c])

                        shape = (f, f)

                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)

    assert(dA_prev.shape == A_prev.shape)

    return dA_prev


np.random.seed(1)
A_prev = np.random.randn(5, 5, 3, 2)
hparameters = {"stride" : 1, "f": 2}
A, cache = pool_forward(A_prev, hparameters)
dA = np.random.randn(5, 4, 2, 2)
dA_prev = pool_backward(dA, cache, mode = "max")
print("mode = max")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1])  
print()
dA_prev = pool_backward(dA, cache, mode = "average")
print("mode = average")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1])