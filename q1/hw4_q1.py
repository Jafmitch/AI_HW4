"""
- File: hw4_q1.py
- Author: Jason A. F. Mitchell, Jarod Osborn, and Eric Tulowetzke
- Summary: This is the main module for homework 4 problem 1.
"""

import graph_wrapper as gw
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import helper_functions as hf

TILE_SIZE = 20
TEST_SIZE = 40  # Number of sample randomly removed for testing set
HIDDEN_LAYER = 200  # Number of dimensions a hidden layer will have
END = 0.01  # And condition for training loop
LEARNING_RATE = 1e-3
TRAIL = 100


def main():
    """
    The main function of this module.
    """
    result = []
    image = heatFlowCA(5)
    for i in range(TRAIL):
        print('Trail', i)
        train_input, train_output, test_input, test_output = sepDate(image=image)
        train_model, cost = torchTrain(train_input, train_output)
        result.append(torchTest(train_model, test_input, test_output))
        print(result[i])
        graphCosts(cost, i)
    graphCorrectAnswers(result)
    result = np.array(result)
    print('mean', np.mean(result), 'std', np.std(result))


def heatFlowCA(scale):
    """
    This function creates an cellular automaton of heat flow on a square
    Args:
        scale: int that help determines how long CA evaluates for
    Returns:
        np.ndarray: 2D array of what heat is on TILE_SIZE by TILE_SIZE plate
    """
    image = np.zeros((TILE_SIZE, TILE_SIZE))
    buffer = np.zeros((TILE_SIZE, TILE_SIZE))
    for i in range(1, TILE_SIZE - 1):
        image[i, TILE_SIZE - 1] = i * (TILE_SIZE - 1 - i)

    for t in range(scale * TILE_SIZE):
        for i in range(1, TILE_SIZE - 1):
            for j in range(1, TILE_SIZE - 1):
                buffer[i, j] = (image[i - 1, j] + image[i + 1, j] + image[i, j - 1] + image[i, j + 1]) / 4.0
        for i in range(1, TILE_SIZE - 1):
            for j in range(1, TILE_SIZE - 1):
                image[i, j] = buffer[i, j]

    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    plt.savefig("images/heatflow2.jpg")
    plt.clf()
    return image


def sepDate(image):
    """
    This function Sports the CA results into inputs and outputs,
    then splits up the data more into training versus testing data.
    Args:
        np.ndarray: 2D array of what heat is on TILE_SIZE by TILE_SIZE plate
    Returns:
        torch.Tensor: values is the training input, which is a 2 by 1 matrix of x and y coordinates
        torch.Tensor: known is the training output
        torch.Tensor: test_input is the testing input, which is a 2 by 1 matrix of x and y coordinates
        torch.Tensor: test_output is the testing output
    """
    # break up date to inputs and outputs
    input_data = []
    output_data = []
    for i in range(0, TILE_SIZE):
        for j in range(0, TILE_SIZE):
            tmp = np.matrix([[i], [j]], dtype=float)
            input_data.append(tmp)
            output_data.append(image[i, j])

    # randomly break up data into train and test data
    stop = len(output_data)
    test_sample = np.random.choice(range(stop), TEST_SIZE, replace=False)
    train_input = []
    train_output = []
    test_input = []
    test_output = []
    for i in range(0, stop):
        if i in test_sample:
            test_input.append(input_data[i])
            test_output.append(output_data[i])
        else:
            train_input.append(input_data[i])
            train_output.append(output_data[i])
    train_input = np.array(train_input)
    train_output = hf.T1D(np.array(train_output))
    test_input = np.array(test_input)
    test_output = hf.T1D(np.array(test_output))
    train_input = torch.Tensor(train_input)
    train_output = torch.Tensor(train_output)
    test_input = torch.Tensor(test_input)
    test_output = torch.Tensor(test_output)
    return train_input, train_output, test_input, test_output


def torchTrain(train_input, train_output):
    """
    This function builds the ANN model,
    along with also training the model using training data that is passed in
    Args:
        train_input: pytorch Tensor of N X 2 of input of x and y values
        train_output: pytorch Tensor of N X 1 of know outputs
            N = TILE_SIZE^2 - TEST_SIZE
    Returns:
            model: pytorch trained ANN
            cost: python array of loss value for each iteration of the training loop
     """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=2, out_features=HIDDEN_LAYER),
        nn.Sigmoid(),
        nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER),
        nn.Sigmoid(),
        nn.Linear(HIDDEN_LAYER, 1),
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss = 1e+5
    cost = []
    while loss > END:
        # Forward propagation
        pred = model(train_input)
        loss = loss_fn(pred, train_output)
        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cost.append(loss.detach())
    return model, cost


def torchTest(model, test_input, test_output):
    """
    This function tests how good the artificial neural network is compared to the testing data.
    It will print the percentage of correct answers.
    Args:
        model: pytorch trained ANN
        test_input: pytorch Tensor of TEST_SIZE X 2 of input of x and y values
        test_output: pytorch Tensor of TEST_SIZE X 1 of know outputs
    Returns:
        float: number of correct values calculated by the ANN
    """
    check = 0
    pred = model(test_input)
    for j in range(TEST_SIZE):
        guess = torch.round(pred[j])
        answer = torch.round(test_output[j])
        if guess == answer:
            check += 1
    print("Number right", check)
    return check / TEST_SIZE


def graphCorrectAnswers(percentCorrectAnswers):
    """
    Creates a graph of the percent correct answers per trial.
    Args:
        percentCorrectAnswers (list): list of percentCorrectAnswers per trial.
    """
    plot = gw.Plot()
    for i in range(len(percentCorrectAnswers)):
        plot.addPoint(i, percentCorrectAnswers[i], "o", "purple")

    plot.label("Percent Answers Correct per Trial",
               "Trial", "Percent of Answers Correct ")
    plot.save("correct_answers.jpg")
    plot.freeze()


def graphCosts(costs, number=0):
    """
    Creates a graph of the cost function results per iteration.
    Args:
        costs (list): List of cost function results per iteration.
        number (int): Number assigned to graph when doing multiple
                      trials. Default is 0.
    """
    plot = gw.Plot()
    plot.pointplot(range(len(costs)), costs, "blue")

    plot.label("Cost per Iteration",
               "Iteration Number", "Cost")
    plot.save("cost_per_iteration_trial" + str(number) +".jpg")
    plot.freeze()


if __name__ == "__main__":
    main()
