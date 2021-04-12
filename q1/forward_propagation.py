"""
- File: forward_propagation.py
- Author: Eric Tulowetzke
- Summary: This module contains a function for forward propagation of a artificial neural network
"""

import numpy as np
import activation_functions as a


def forward_network(neuron_layer_array, know, first=True):
    """
    This function accepts an numpy array of instances of the neuron layer class.
    Along with the number of layers that the ANN supposed to have.
    It then runs through forward propagation for all layers, and updates values in the layers.

    Args:
        neuron_layer_array: numpy array of neuron layer instances.
        know(numpy array): that has the know outcome

    Returns:
         Returns the square of prediction minus know value"""
    layer = neuron_layer_array.shape[0]
    for l in range(0, layer):
        if first is True:
            neuron_layer_array[l].z = np.dot(neuron_layer_array[l].w, neuron_layer_array[l].input_value)
            neuron_layer_array[l].a = a.activation(neuron_layer_array[l].z)
            first = False
        elif l == (layer-1):
            neuron_layer_array[l].input_value = neuron_layer_array[l - 1].a
            neuron_layer_array[l].z = np.dot(neuron_layer_array[l].w, neuron_layer_array[l].input_value)
            neuron_layer_array[l].a = neuron_layer_array[l].z
        else:
            neuron_layer_array[l].input_value = neuron_layer_array[l-1].a
            neuron_layer_array[l].z = np.dot(neuron_layer_array[l].w, neuron_layer_array[l].input_value)
            neuron_layer_array[l].a = a.activation(neuron_layer_array[l].z)
    return (np.square(np.subtract(neuron_layer_array[layer - 1].a, know))).sum()

