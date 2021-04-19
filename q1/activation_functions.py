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
    return 1 / (1 + np.exp(-xArray))

def activationPrime(xArray):
    """
    Calculates y = sigmoid'(x)

    Args:
        xArray (np.ndarray): Array of x values.

    Returns:
        np.ndarray: Array of y values in nx1 form.
    """
    f = activation(xArray=xArray)
    f2 = np.subtract(1, f)
    return np.multiply(f, f2)