import numpy  as np
import pandas as pd
from random import random, seed
from math   import exp

def normalize(dataset):
    '''
        Normalizes the data provided in dataset using min-max method.
    '''
    vector_min = []
    vector_max = []

    #this is the normalized version of X
    normalizedDataset = dataset

    n_columns = dataset.shape[1]
    for i in range(n_columns - 1):
        m = np.min(dataset[i])
        M = np.max(dataset[i])
        vector_min.append(m)
        vector_max.append(M)
        normalizedDataset[i]  = np.subtract(normalizedDataset[i], m)
        normalizedDataset[i]  = np.divide(normalizedDataset[i],M - m)

    return normalizedDataset, vector_min, vector_max

def readData(trainset):

    dataset = pd.read_csv(trainset,delim_whitespace = True,header = None,index_col = False)
    dataset, mean, std = normalize(dataset)

    #fix the dataset as an array of [x1, x2, x3,..., y]
    aux_dataset = []
    for i in range(len(dataset)):
        row = dataset.iloc[i]
        aux_row = []
        for j in range(len(row)):
            aux_row.append(row[j])

        aux_dataset.append(aux_row)

    dataset = aux_dataset

    return dataset

# Initializing the network
def initialize(n_inputs, n_hidden, n_outputs):
    network = list()

    hidden_layer = []
    for i in range(n_hidden):
        weights  = []
        for j in range(n_inputs + 1):
            weights.append(random())
        aux_dict = {'weights': weights}
        hidden_layer.append(aux_dict)

    network.append(hidden_layer)

    outputlayer = []
    for i in range(n_outputs):
        weights  = []
        for j in range(n_hidden + 1):
            weights.append(random())
        aux_dict = {'weights': weights}
        outputlayer.append(aux_dict)

    network.append(outputlayer)
    return network

def activate_neuron(weights, inputs):
    activation = weights[-1]          #last weight is the bias
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation

def transfer(activation):
    return 1.0/(1.0 + exp(-activation))

def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate_neuron(neuron['weights'],inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# this is for backpropagation
def transfer_derivate(output):
    return output*(1.0 - output)

def backpropagation(network, expected):

    #calculate errors
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivate(neuron['output'])

def update_weights(network, row, alpha):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = []
            for neuron in network[i-1]:
                inputs.append(neuron['output'])
        for neuron in network[i]:
            for j in range(len(inputs)):

                neuron['weights'][j] += alpha*neuron['delta']*inputs[j]
            neuron['weights'][-1] += alpha*neuron['delta']

def train(network, trainset, alpha, n_epoch, n_outputs):

    # we save cost vs iterations for the training set
    iter_vs_cost = [[],[]]

    for epoch in range(n_epoch):
        cost = 0
        for row in trainset:
            output   = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1
            cost += sum([(expected[i] - output[i])**2 for i in range(len(expected))]) #this calculates the cost function
            backpropagation(network, expected)
            update_weights(network, row, alpha)

        iter_vs_cost[0].append(epoch)
        iter_vs_cost[1].append(cost)
            # print('> epoch=%d, alpha=%.3f, error=%.10f' % (epoch, alpha, cost))

    #the last error in the cosv_vs_iterations[0] is the min
    return iter_vs_cost

def predict(network, row):
    output = forward_propagate(network, row)
    return output.index(max(output))

def calculate_predictions(network, testset):
    predictedset = []
    expected_vs_predicted = []
    for row in testset:
        prediction = predict(network, row)
        # print('Expected=%d, predicted=%d'%(row[-1],prediction))
        expected_vs_predicted.append([row[-1],prediction])
        predictedset.append([row[0],row[1],prediction])

    return predictedset, expected_vs_predicted

def calculate_errors(results):

    total_error     = 0
    false_positives = 0
    false_negatives = 0


    #this is the total square errors
    total_error += sum([(result[0] - result[1])**2 for result in results])

    for result in results:
        #false positive
        if result[0] == 0 and result[1] == 1:
            false_positives += 1
        #false negative
        if result[0] == 1 and result[1] == 0:
            false_negatives += 1

    return total_error, false_positives, false_negatives
