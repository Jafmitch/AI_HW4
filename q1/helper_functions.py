"""
- File: helper_functions.py
- Author: Jason A. F. Mitchell
- Summary: This module is for miscellaneous helper functions useful to this
           program.
"""

import numpy as np

def T1D(array1d):
    """
    Turns a 1d numpy array into a 2d numpy array and transposes it into a 
    column matrix.

    Args:
        array1d (np.ndarray): 1d numpy array

    Returns:
        np.ndarray: 2d numpy array transposed into a column matrix
    """
    return to2D(array1d).T


def to2D(array1d):
    """
    Turns a 1d numpy array into a 2d numpy array. Returns the array if already
    2d.

    Args:
        array1d (np.ndarray): 1d numpy array

    Returns:
        np.ndarray: 2d numpy array transposed into a column matrix
    """
    if len(array1d.shape) == 2:
        return array1d
    else:
        return array1d[np.newaxis]
