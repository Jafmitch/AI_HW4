"""
File: JuliaSet.py
Author: Jarod Osborn
Description: This program generates an n-dimensional Julia set from a given complex number c.
Class: CSC 468 - Advanced Topics in AI.
"""

import uuid
import numpy as np

from PIL import Image

IMG_SIZE = (4000, 4000)


def __findR(n: int, cx: float, cy: float) -> int:
    """
    Finds the smallest radius R (>= 2) that fulfills the condition:
     R^n is greater than or equal to the square root of (cx^2 + cy^2)

    :param n: The n dimension of the set.
    :param cx: The x component of the complex number c.
    :param cy: The y component of the complex number c.
    :return: The smallest radius R that fulfills the condition.
    """
    R = 2

    while pow(R, n) < np.sqrt(cx * cx + cy * cy):
        R += 1

    return R


def __normalizeGrid(grid: np.ndarray) -> np.ndarray:
    """
    Normalizes the G, B, and A values of a image grid by finding the greatest G, B, and A values
    and then setting all G, B, and A values to a proportion of 255 based on the respective max values.

    :param grid: The image grid to normalize
    :return: The normalized image grid.
    """
    maxG = 0
    maxB = 0
    maxA = 0
    for y in range(IMG_SIZE[1]):
        for x in range(IMG_SIZE[0]):
            # We don't look for max R, it's always 51.

            # Find Max G value.
            if grid[x, y, 1] > maxG:
                maxG = grid[x, y, 1]

            # Find Max B value.
            if grid[x, y, 2] > maxB:
                maxB = grid[x, y, 2]

            # Find Max Alpha value.
            if grid[x, y, 3] > maxA:
                maxA = grid[x, y, 3]

    # Normalize all G, B, and A values to a proportion of 255 based on respective max.
    for y in range(IMG_SIZE[1]):
        if y % (IMG_SIZE[1] / 20) == 0:
            percentage = "{:.1f}".format((y / IMG_SIZE[1]) * 100)
            print(f"Normalization: {percentage}%")

        for x in range(IMG_SIZE[0]):
            grid[x, y, 1] = (grid[x, y, 1] / maxG) * 255
            grid[x, y, 2] = (grid[x, y, 2] / maxB) * 255
            grid[x, y, 3] = (grid[x, y, 3] / maxA) * 255

    return grid


def createJuliaSet(n: int = 2, cx: float = -0.760, cy: float = 0.080, maxIter: int = 255):
    """
    Creates an n-dimensional Julia set image grid.

    :param n: The desired n dimension of the Julia set.
    :param cx: The x component of the complex number c.
    :param cy: The y component of the complex number c.
    :param maxIter: The maximum number of iterations (default = 255)
    :return: An image grid containing the Julia set.
    """
    grid = np.zeros(shape=(IMG_SIZE[0], IMG_SIZE[1], 4), dtype=np.uint8)

    R = __findR(n, cx, cy)

    # print(R)

    for y in range(IMG_SIZE[1]):
        # A little checker to ensure progress.
        if y % (IMG_SIZE[1] / 20) == 0:
            percentage = "{:.1f}".format((y / IMG_SIZE[1]) * 100)
            print(f"Creation: {percentage}%")

        for x in range(IMG_SIZE[0]):
            # Normalize zx and zy to values inside the grid.
            zx = (x / IMG_SIZE[0]) * (2 * R) - R
            zy = (y / IMG_SIZE[1]) * (2 * R) - R

            itr = 0

            # I had this working with the normal n=2 Julia sets, but
            #  wanted to have some fun, so I did some research on Julia multi-sets
            while zx * zx + zy * zy < pow(R, 2) and itr < maxIter:
                # Get our magnitude/length/etc to the n/2 power
                mag = pow((zx * zx + zy * zy), (n / 2))

                # Generate our new zx and zy using cos/sin and arctan2.
                xTemp = mag * np.cos(n * np.arctan2(zy, zx)) + cx
                zy = mag * np.sin(n * np.arctan2(zy, zx)) + cy
                zx = xTemp

                itr += 1

            # Reached max iterations, set pixel to transparency
            if itr == maxIter:
                grid[x, y, 0] = 0
                grid[x, y, 1] = 0
                grid[x, y, 2] = 0
                grid[x, y, 3] = 0
            # Escaped radius, set pixel color with iteration based alpha.
            else:
                grid[x, y, 0] = 51  # RED
                grid[x, y, 1] = abs(zx)  # GREEN
                grid[x, y, 2] = abs(zy)  # BLUE

                grid[x, y, 3] = itr * n  # ALPHA

                # If iterations were over 100/255 of max iterations, set alpha to max.
                if itr >= (100 / 255) * maxIter:
                    grid[x, y, 3] = 255

    return __normalizeGrid(grid)


if __name__ == '__main__':
    jArr = createJuliaSet(n=3, cx=-0.21, cy=0.97, maxIter=500)
    im = Image.fromarray(jArr).convert('RGBA')
    uid = str(uuid.uuid4())
    im.save(f"JuliaRound_{uid}.png")
    im.show()
