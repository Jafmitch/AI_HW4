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

TILE_SIZE = 10
TEST_FRAC = 0.1
HIDDEN_DIM = 5
END = 1
INPUT_DIM = 2
LEARNING_RATE = bp.LEARNING_RATE
N_LAYER = 4  # number of Neuron layers
OUTPUT_DIM = 1


def main():
    """
    The main function of this module.
    """
    ann = buildPerceptron(N_LAYER, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    image = heatFlowCA()
    train_input, train_output, test_input, test_output = sepDate(image=image)
    trainANN(ann, train_output, train_input)
    print(testANN(ann, test_output, test_input))


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


def heatFlowCA():
    """
        This function creates an cellular automaton of heat flow on a square

        Returns:
            np.ndarray: 2D array of what heat is on TILE_SIZE by TILE_SIZE plate
        """
    image = np.zeros((TILE_SIZE, TILE_SIZE))
    buffer = np.zeros((TILE_SIZE, TILE_SIZE))
    for i in range(1, TILE_SIZE - 1):
        image[i, TILE_SIZE - 1] = i * (TILE_SIZE - 1 - i)

    for t in range(5 * TILE_SIZE):
        for i in range(1, TILE_SIZE - 1):
            for j in range(1, TILE_SIZE - 1):
                buffer[i, j] = (image[i - 1, j] + image[i + 1, j] + image[i, j - 1] + image[i, j + 1]) / 4.0
        for i in range(1, TILE_SIZE - 1):
            for j in range(1, TILE_SIZE - 1):
                image[i, j] = buffer[i, j]
    return image


def sepDate(image):
    """
        This function Sports the CA results into inputs and outputs,
        then splits up the data more into training versus testing data.

        Args:
            np.ndarray: 2D array of what heat is on TILE_SIZE by TILE_SIZE plate

        Returns:
            np.ndarray: values is the training input, which is a 2 by 1 matrix of x and y coordinates
            np.ndarray: known is the training output
            np.ndarray: test_input is the testing input, which is a 2 by 1 matrix of x and y coordinates
            np.ndarray: test_output is the testing output
    """
    values = []
    known = np.array([])
    for i in range(0, TILE_SIZE):
        for j in range(0, TILE_SIZE):
            tmp = np.matrix([[i], [j]])
            values.append(tmp)
            del tmp
            known = np.append(known, image[i, j])

    values = np.array(values)
    stop = known.size + 1
    test_size = round(stop * TEST_FRAC)
    test_sample = np.random.randint(0, stop, test_size)
    test_input = []
    test_output = np.array([])
    test_output = np.append(test_output, known[test_sample])
    known = np.delete(known, test_sample)
    for i in test_sample:
        test_input.append(values[i])
    values = np.delete(values, test_sample)
    test_input = np.array(test_input)
    return values, known, test_input, test_output


def trainANN(ann, know, values):
    """
    This function trains the neural network by feeding it training data and
    adjusting its weights according to a calculated gradient.

    Args:
        ann (np.ndarray): An artificial neural network represented by an array
                          or NeuronLayer objects
    Returns:
        list: list of the cost of each iteration
    """
    costs = []
    # Feed in each input value into both forward propagation and back propagatio
    batch = know.size
    loss = batch
    past = batch + 1
    k = 0
    knockOut = 0
    while loss > END and knockOut < 15:
        loss_arr = np.array([])
        for i in range(batch):
            ann[0].input_value = values[i]
        # Collect the square difference of each pair
            sqdiff = fp.forward_network(ann, know[i])
            loss_arr = np.append(loss_arr, sqdiff)
            bp.backprop(ann, know[i])
        # Finish Average and the sum gradients and edit Weights
        for i in range(N_LAYER):
            ann[i].w -= ann[i].dCdw_sum/batch * LEARNING_RATE
            ann[i].zero_out()
        loss = loss_arr.sum() / batch
        costs.append(loss)
        if k % 1000 == 0:
            print(k, loss)
            if past < loss:
                knockOut += 1
            else:
                knockOut = 0
                past = loss
        k += 1
        del loss_arr
    print("final lost", k, loss)
    return costs


def testANN(ann, know, values):
    """
        This function tests how good the artificial neural network is compared to the testing data.
        It will print the percentage of correct answers.

        Args:
            ann(array of Neuron layer class):
        Returns:
            float: number of correct values calculated by the ANN
    """
    total = know.size
    check = 0
    for j in range(total):
        ann[0].input_value = values[j]
        fp.forward_network(ann, know[j])
        guess = np.around(ann[N_LAYER - 1].a[0])
        pred = round(know[j])
        #print(j, "guess", guess)
        #print(j, "know", pred)
        if guess == pred:
            check += 1
    print("Number right", check)
    # print("Total data points", layer)
    # print("testpercentage ", check / layer)
    return check / total

if __name__ == "__main__":
    main()
