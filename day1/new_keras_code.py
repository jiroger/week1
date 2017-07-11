import numpy as np


def convert_kernel(kernel, dim_ordering):
    """ Converts a kernel matrix (Numpy array) from Theano format to
        TensorFlow format, and vice-versa.

        Parameters
        ----------
        kernel : numpy.ndarray
            Theano or TensorFlow-style convolution kernel
        dim_ordering : str
            Theano-style: 'th'
            TensorFlow-style: 'tf'

        If Theano-style kernel, the kernel shape is: (C, F, H, W) or (C, F, H, W, D)
        If TensorFlow-style kernel, the kernel shape is: (H, W, C, F) or (H, W, D, C, F)

        Reverse the height (H), width (W), and (if present) depth (D) axes.
        C & F axes should not be changed.

        C : number of channels of data
        F : number of convolution filters
        H : height dimension
        W : width dimension
        D : depth dimension """
    
    # flip all axes
    # make a list of slices for the appropriate number of dimensions
    # 
    slices = [slice(None, None, -1) for i in range(kernel.ndim)]
    
    # don't flip C & F
    # updates the C & F indices to be slice(None, None)
    no_flip = (slice(None, None), slice(None, None))
    if dim_ordering == 'th':
        slices[:2] = no_flip  # (C, F, H, W, ..)
    else:
        slices[-2:] = no_flip  #(H, W, .., C, F)

    # make slices a tuple for basic indexing
    slices = tuple(slices)

    return np.copy(kernel[slices])  # why should I make a copy here?


def convert_kernel_fancy(kernel, dim_ordering):
    """ Converts a kernel matrix (Numpy array) from Theano format to
        TensorFlow format, and vice-versa.

        Parameters
        ----------
        kernel : numpy.ndarray
            Theano or TensorFlow-style convolution kernel
        dim_ordering : str
            Theano-style: 'th'
            TensorFlow-style: 'tf'

        If Theano-style kernel, the kernel shape is: (C, F, H, W) or (C, F, H, W, D)
        If TensorFlow-style kernel, the kernel shape is: (H, W, C, F) or (H, W, D, C, F)

        Reverse the height (H), width (W), and (if present) depth (D) axes.
        C & F axes should not be changed.

        C : number of channels of data
        F : number of convolution filters
        H : height dimension
        W : width dimension
        D : depth dimension """
    slices = [slice(None, None, -1) for i in range(kernel.ndim)]
    no_flip = (slice(None, None), slice(None, None))
    index = slice(None, 2) if dim_ordering == 'th' else slice(-2, None)
    slices[index] = no_flip  
    return np.copy(kernel[tuple(slices)])  # why should I make a copy here?
