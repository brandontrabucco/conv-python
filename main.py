"""

Author: Brandon Trabucco.
Creation Date: 2017.06.28.
Program Name: Convolutional Neural Network.
Program Description:
    This program implements a convolutional neural network as a single layer.
    The convolution algorithm extends the feed forward neural network with locally receptive fields and shared trainable parameters.
    This algorithm adds translation invariance to the feed forward neural network, which improves pattern recognition.


Version Name: Fully Implemented Forward & Backward Propagation
Version Number: v0.1-alpha
Version Description:
    This version contains a fully functional convolution algorithm, which may be connected to other types of neural network layers to form a full network.
    More robust unit testing is needed in order to find bugs, and prove consistency.
    In the future, a pooling layer will be added in for more algorithmic flexibility and to improve pattern learning.

"""


"""
The NumPy library is used extensively in this python application.
This library provides a comprehensive set of linear algebra classes and functions.
"""
import numpy as np

"""
The im2col file was taken from the CS231n course from Stanford University.
This file provides a fast method of converting a four-dimensional input matrix into a two dimensional matrix, where each column is a single instance of a sliding receptive field.
"""
from im2col import im2col_indices, col2im_indices


"""
This class serves as a namespace for this implementation of the convolution algorithm
"""
class conv:

    """
    This algorithm uses the softplus function as a Rectified Linear Unit, built into the convolution layer.
    """
    def activate(x):
        return np.log(1 + np.exp(x))

    """
    The derivative of the softplus function is the standard logistic function
    """
    def aprime(x):
        return 1 / (1 + np.exp(-x))

    """
    This class is a wrapper for the convolution cell below. 
    This algorithm implements standard backpropogation with gradient descent.
    Interact directly with this class, and not the convolution cell.
    """
    class layer:

        """
        This will reset the the convolutional neural network for another training batch.
        The trainable parameters are preserved.
        """
        def reset(self):
            self._cell.reset()

        """
        This function will create the neural network convolution layer, and initialize the trainable parameters.
        """
        def __init__(self, params):
            self.alpha = params['alpha']
            self._cell = conv.cell(params)
            self.reset()

        """
        This functions requires an input, and passes such through the convolution layer.
        The output of this function may be connected to other neural network layers.
        """
        def forward(self, stimulus):
            return self._cell.forward(stimulus)

        """
        This functions will update the trainable parameters of the convolution layer.
        A weighted error will be passed out of this function.
        """
        def backward(self, delta):
            return self._cell.backward(delta, self.alpha)

    """
    This class contains the computation for the convolution algorithm.
    This class contains all trainable parameters, and computer all partial derivatives.
    """
    class cell:

        """
        This function will set the value of the weight and bias matrices to random values according to a normal distribution with mean 0 and standard deviation 1.
        """
        def create(self):
            self.reset()
            self.filters = np.random.normal(0, 1, (self.numFilters, self.filterChannel, self.filterHeight, self.filterWidth))
            self.bias = np.random.normal(0, 1, (self.numFilters, 1))
            self.outHeight = int((self.inputHeight + 2 * self.padding - self.filterHeight) / self.stride + 1)
            self.outWidth = int((self.inputWidth + 2 * self.padding - self.filterWidth) / self.stride + 1)

        """
        This function will reset the saved input to this algorithm.
        """
        def reset(self):
            self.stimulus = np.zeros((1, self.inputWidth))

        """
        This method will create the convolution layer, and read a set of layer parameters.
        """
        def __init__(self, params):
            self.numInput = params['numInput']
            self.inputChannel = params['inputChannel']
            self.inputHeight = params['inputHeight']
            self.inputWidth = params['inputWidth']
            self.numFilters = params['numFilters']
            self.filterChannel = params['filterChannel']
            self.filterHeight = params['filterHeight']
            self.filterWidth = params['filterWidth']
            self.padding = params['padding']
            self.stride = params['stride']
            self.create()

        """
        This function performs the linear algebra math operation for the forward pass of the convolution layer
        The convolution is performed as a single matrix multiplication using the im2col method described in Stanford University course CS231n.
        """
        def forward(self, stimulus):
            self.stimulus = stimulus
            stimulusCols = im2col_indices(self.stimulus, self.filterHeight, self.filterWidth, self.padding, self.stride)
            filterCols = self.filters.reshape((self.numFilters, -1))
            biasCols = self.bias.repeat(self.outHeight * self.outWidth, axis=1)
            activationCols = conv.activate(np.dot(filterCols, stimulusCols) + biasCols)
            self.activation = activationCols.reshape(self.numFilters, self.outHeight, self.outWidth, self.numInput).transpose(3, 0, 1, 2)
            return self.activation

        """
        This function will compute updates to the trainable parameters in this network.
        Partial derivatives are calculated as a convolution, using the col2im method from Stanford University course CS231n
        """
        def backward(self, delta, alpha):
            stimulusCols = im2col_indices(self.stimulus, self.filterHeight, self.filterWidth, self.padding, self.stride)
            filterCols = self.filters.reshape((self.numFilters, -1))
            activationCols = self.activation.transpose(1, 2, 3, 0).reshape(self.numFilters, -1)
            deltaCols = delta.transpose(1, 2, 3, 0).reshape(self.numFilters, -1) * conv.aprime(activationCols)
            inputPartial = col2im_indices(np.dot(deltaCols.transpose(1, 0), filterCols), self.stimulus.shape, self.filterHeight, self.filterWidth, self.padding, self.stride)
            self.filters -= alpha * np.dot(deltaCols, stimulusCols.transpose(1, 0)).reshape(self.filters.shape)
            self.bias -= alpha * np.sum(deltaCols, axis=1).reshape(self.bias.shape)
            return inputPartial

"""
This function serves as the entry point for this python applicaion.
The convolutio layer created above is created, and tested on fake data.
"""
def main():
    layer = conv.layer({'numInput': 1, 'inputChannel': 1, 'inputHeight': 10, 'inputWidth': 10, 'numFilters': 1, 'filterChannel': 1, 'filterHeight': 3, 'filterWidth': 3, 'padding': 1, 'stride': 1, 'alpha': 0.01})
    print(layer.forward(np.ones((1, 1, 10, 10))).shape)
    print(layer.backward(np.zeros((1, 1, 10, 10))).shape)

main()