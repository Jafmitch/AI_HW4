"""
- File: hw4_q1.py
- Author: Jason A. F. Mitchell, Jarod Osborn, and Eric Tulowetzke
- Summary: This is the main module for homework 4 problem 1.
"""
import backward_propagation as bp
import forward_propagation as fp
import graph_wrapper as gw
import neuron_layer as nl
import numpy as np


def main():
    """
    The main function of this module.
    """
    print("Does nothing yet")


def buildPerceptron(layers, i, h, o):
    """
    This function creates an artificial neural network multi-layer perceptron
    using the NeuronLayer object.

    Args:
        layers (int): number of layers in the perceptron
        n ([type]): n dimension of the perceptron
        m ([type]): m dimension of the perceptron

    Returns:
        np.ndarray: array of NeuronLayers representing the perceptron
    """
    perceptron = np.array([])
    for l in range(layers):
        if l == 0:
            tmp = nl.NeuronLayer(h, i)
        elif l == (layers - 1):
            tmp = nl.NeuronLayer(o, h)
        else:
            tmp = nl.NeuronLayer(h, h)
        perceptron = np.append(perceptron, tmp)
    return perceptron


if __name__ == "__main__":
    main()
