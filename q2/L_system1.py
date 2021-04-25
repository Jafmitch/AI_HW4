"""
- File: L_system1.py
- Author: Jason A. F. Mitchell
- Summary: This module generates Lindenmayer System fractal images.
"""
import image_generator as ig
import numpy as np

IMAGE_LENGTH = 2000

# colors
AQUA = [0, 255, 255]
BLACK = [0, 0, 0]
BLUE = [70, 48, 255]
GREEN = [0, 255, 0]
GREY = [99, 99, 99]
PURPLE = [106, 0, 128]
RED = [255, 0, 0]
WHITE = [255, 255, 255]


def createFractal_LindenmayerSystem():
    """
    This function generates a Lindenmayer System based on a series of pixel 
    colors and uses those pixels to create an image. For each row of the image,
    a new set of pixel values are generated based on certain production rules
    as well as the previous values. The generated values are then used to color
    the image. Once the image is created it is saved as a jpg.

    """
    image = np.zeros([IMAGE_LENGTH, IMAGE_LENGTH, 3], dtype=np.uint8)
    pixelSet = initializeLSystem([PURPLE])
    for j in range(IMAGE_LENGTH):
        print(j)
        pixelSet = getNextSet(pixelSet)
        for i in range(IMAGE_LENGTH):
            if i < len(pixelSet):
                image[i][j] = pixelSet[i]
            else:
                image[i][j] = BLACK
    ig.saveImageFromArray(image, "l_system1.png")


def getNextSet(previousSet):
    """
    Creates a new pixel set based on a previous pixelSet based on the following rules:
    - AQUA -> BLUE
    - BLUE -> GREY, PURPLE
    - PURPLE -> BLUE, GREY
    - GREY -> AQUA, PURPLE

    Args:
        previousSet (list): previous pixel set in the algorithm

    Returns:
        list: next pixel set in the algorithm
    """
    nextSet = []
    for item in previousSet:
        if item == AQUA:
            nextSet.append(BLUE)
        elif item == BLUE:
            nextSet.append(GREY)
            nextSet.append(PURPLE)
        elif item == PURPLE:
            nextSet.append(BLUE)
            nextSet.append(GREY)
        elif item == GREY:
            nextSet.append(AQUA)
            nextSet.append(PURPLE)
        while len(nextSet) > IMAGE_LENGTH:
            nextSet.pop()
    return nextSet


def initializeLSystem(axiom):
    """
    Initializes Lindenmayer System by generating new pixel sets until the set
    is the same length as a row in the image being generated.

    Args:
        axiom (list): initial pixel set of L-system

    Returns:
        list: pixel set based on the axiom that is the same length as the image
    """
    pixelSet = axiom
    while len(pixelSet) < IMAGE_LENGTH:
        pixelSet = getNextSet(pixelSet)
    return pixelSet


if __name__ == "__main__":
    createFractal_LindenmayerSystem()
