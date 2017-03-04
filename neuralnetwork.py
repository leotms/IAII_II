from random import random, seed
from math   import exp

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
    for epoch in range(n_epoch):
        sum_eror = 0
        for row in trainset:
            output   = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_eror += sum([(expected[i] - output[i])**2 for i in range(len(expected))])
            backpropagation(network, expected)
            update_weights(network, row, alpha)
        print('> epoch=%d, alpha=%.3f, error=%.10f' % (epoch, alpha, sum_eror))

def predict(network, row):
    output = forward_propagate(network, row)
    return output.index(max(output))

# seed(1)
# network = initialize(2,1,2)
# for layer in network:
#     print(layer)
#
# row = [1,0,None]
# output = forward_propagate(network, row)
#
# print('Output', output)
#
# print('After backpropagation')
# expected = [0,1]
# backpropagation(network, expected)
# for layer in network:
#     print(layer)

seed(1)
dataset = [[2.7810836,2.550537003,0],
  [1.465489372,2.362125076,0],
  [3.396561688,4.400293529,0],
  [1.38807019,1.850220317,0],
  [3.06407232,3.005305973,0],
  [7.627531214,2.759262235,1],
  [5.332441248,2.088626775,1],
  [6.922596716,1.77106367,1],
  [8.675418651,-0.242068655,1],
  [7.673756466,3.508563011,1]]

n_inputs = len(dataset[0]) - 1
n_outputs  = len(set(row[-1] for row in dataset))
network = initialize(n_inputs, 10, n_outputs)
train(network, dataset, 0.1, 1000, n_outputs)
for layer in network:
    print(layer)

for row in dataset:
    prediction = predict(network, row)
    print('Expected=%d, predicted=%d'%(row[-1],prediction))
