"""
- File: neuron_layer.py
- Author: Eric Tulowetzke
- Summary: This module contains a class(struct) of a layer of neurons of an ANN.
"""
import numpy as np

class NeuronLayer:
    """
    This class acts as a Structure that contains the information for one layer of an ANN.

    Attributes:
        input_value(1D numpy array): Holds the input values of the ANN.
        w(m by n numpy array): This holds the values of weights for a layer in a matrix like form.
        z(1D numpy array): Holds the values of W*x for each layer of the ANN.
        a(1D numpy array): Holds the values of z after an activation function has been applied.
        dCdz(1D numpy array): Is the gradient of the cost function with respect to change in z.
        dCdw(1D numpy array): Is the gradient of the cost function with respect to change in w.
    """
    def __init__(self, m_dim, n_dim):
        """
        Constructor for class.
        It will random Initialization of the w when created.

        Args:
              m_dim(int): number of rows in weight matrix
              n_dim(int): number of columns in weight matrix

        """
        self.input_value = np.array([])
        self.w = np.random.uniform(0,1, (m_dim, n_dim))
        self.z = np.array([])
        self.a = np.array([])
        self.dCdz = np.array([])
        self.dCdw = np.array([])
        self.dCdw_sum = np.zeros((m_dim, n_dim)) #hold sum of dCdw per batch
        self.m_dim = m_dim
        self.n_dim = n_dim

    #use to reset sum of dCdw
    def zero_out(self):
        self.dCdw_sum = np.zeros((self.m_dim, self.n_dim))