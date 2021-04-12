"""
- File: activation_functions.py
- Author: Jason A. F. Mitchell
- Summary: This module has some common activation functions for neural networks.
"""

import numpy as np

def activation(xArray):
    """
    Calculates y = sigmoid(x)

    Args:
        xArray (ndarray): Array of x values.

    Returns:
        ndarray: Array of y values.
    """
    s = xArray.shape
    one_matrix = np.ones(s)
    e = np.exp(-xArray)
    don = np.add(one_matrix, e)
    return 1 / don

def activationPrime(xArray):
    """
    Calculates y = sigmoid'(x)

    Args:
        xArray (np.ndarray): Array of x values.

    Returns:
        np.ndarray: Array of y values in nx1 form.
    """
    f = activation(xArray=xArray)
    s = xArray.shape
    one_matrix = np.ones(s)
    f2 = np.subtract(one_matrix, f)
    return np.multiply(f, f2)