"""
- File: L_system2.py
- Author: Jason A. F. Mitchell
- Summary: This module generates Lindenmayer Systemfractal images.
"""

import image_generator as ig
import numpy as np

IMAGE_LENGTH = 2000
MAX_ITERATIONS = 10000000

# colors
AQUA = [0, 255, 255]
BLACK = [0, 0, 0]
BLUE = [70, 48, 255]
GREEN = [0, 255, 0]
GREY = [99, 99, 99]
PURPLE = [106, 0, 128]
RED = [255, 0, 0]
WHITE = [255, 255, 255]

# directions
NORTH = (0, 1)
NORTH_EAST = (1, 1)
EAST = (1, 0)
SOUTH_EAST = (1, -1)
SOUTH = (0, -1)
SOUTH_WEST = (-1, -1)
WEST = (-1, 0)
NORTH_WEST = (-1, 1)

# rules
DRAW = 0
MOVE = 1
ROTATE = 2

RULE_SET = [
    [[MOVE], [ROTATE, MOVE], [MOVE, DRAW]],
    [[MOVE, MOVE], [ROTATE, MOVE], [MOVE, DRAW]]
]


def createFractal_LindenmayerSystem(number, rules):
    """
    This function generates a Lindenmayer System based on a series of 
    instructions. The program controls a digital marker with a position and 
    direction vector; the instructions determine how the marker is used. When
    the set of actions are completed, a new set is generated using the provided 
    set of production rules. Once the algorithm reaches a certain end condition
    it saves the image generated as a jpg. Here are the instructions the program
    understands:
    - DRAW: color in a pixel at the current position
    - MOVE: change current position based on direction vector
    - ROTATE: change direction vector right by 45 degrees

    Args:
        number (int): number to add to subscript the image name
        rules (list): production rules used to generate instructions
    """
    image = np.zeros([IMAGE_LENGTH, IMAGE_LENGTH, 3], dtype=np.uint8)
    instructions = [DRAW]
    iNum = 0
    direction = NORTH
    position = [int(IMAGE_LENGTH / 2), int(IMAGE_LENGTH / 2)]
    i = 0
    while not endCondition(position, i, MAX_ITERATIONS):
        print(i)
        if iNum >= len(instructions):
            instructions = getNextInstructionSet(instructions, rules)
            iNum = 0
        instruction = instructions[iNum]

        if instruction == DRAW:
            image[position[0]][position[1]] = AQUA
        elif instruction == MOVE:
            position[0] += direction[0]
            position[1] += direction[1]
        elif instruction == ROTATE:
            direction = rotate(direction)
        iNum += 1
        i += 1
    ig.saveImageFromArray(image, "l_system2_" + str(number) + ".png")


def endCondition(position, index, maxIndex):
    """
    This function determines when the algorithm created in 
    createFractal_LindenmayerSystem() will terminate. In the current 
    implementation, termination is based on index and position. If the index
    exceeds a certain amount or the position goes outside the image, this 
    function will return true.

    Args:
        position (list): size 2 list with an x and y value
        index (int): index of algorithm
        maxIndex (int): max index of algorithm

    Returns:
        true when algorithm should terminate
        false otherwise
    """
    return index >= maxIndex or \
        position[0] >= IMAGE_LENGTH or position[1] >= IMAGE_LENGTH or \
        position[0] < 0 or position[1] < 0


def getNextInstructionSet(previousSet, rules):
    """
    Generates a set of instructions based on a set of production rules and 
    another set of similar instructions.

    Args:
        previousSet (list): set of instructions
        rules (list): 2d list representing production rules

    Returns:
        list: new set of instructions
    """
    nextSet = []
    for item in previousSet:
        if item == DRAW:
            for draw_rule in rules[0]:
                if draw_rule > -1:
                    nextSet.append(draw_rule)
        elif item == MOVE:
            for move_rule in rules[1]:
                if move_rule > -1:
                    nextSet.append(move_rule)
        elif item == ROTATE:
            for rotate_rule in rules[2]:
                if rotate_rule > -1:
                    nextSet.append(rotate_rule)
    return nextSet


def rotate(currentDirection):
    """
    This function handles the "rotate" instruction from the 
    createFractal_LindenmayerSystem() function. There are 8 predefined direction
    vectors based on cardinal directions, this function takes in the current 
    direction as a parameter and returns the direction vector that points 45 
    degrees to the right of the previous one.

    Args:
        currentDirection (tuple): tuple representing previous direction vector

    Returns:
        tuple: tuple representing next direction vector
    """
    if currentDirection == NORTH:
        return NORTH_EAST
    elif currentDirection == NORTH_EAST:
        return EAST
    elif currentDirection == EAST:
        return SOUTH_EAST
    elif currentDirection == SOUTH_EAST:
        return SOUTH
    elif currentDirection == SOUTH:
        return SOUTH_WEST
    elif currentDirection == SOUTH_WEST:
        return WEST
    elif currentDirection == WEST:
        return NORTH_WEST
    else:
        return NORTH


if __name__ == "__main__":
    for rule in range(len(RULE_SET)):
        createFractal_LindenmayerSystem(rule, RULE_SET[rule])
