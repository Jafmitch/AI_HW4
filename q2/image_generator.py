"""
- File: image_generator.py
- Author: Jason A. F. Mitchell
- Summary: This module contains an image generating function.
"""
from PIL import Image
import numpy as np

FILE = "images/"


def saveImageFromArray(array, nameOfImage="image.jpg"):
    """
    Turns an np array of tuples into an image using PIL.

    Args:
        array (np.ndarray): np array of tuples representing RGB values
        nameOfImage (str): Name of file image will be saved to. Defaults to "image.jpg".
    """
    im = Image.fromarray(array, "RGB")
    im.save(FILE + nameOfImage)
