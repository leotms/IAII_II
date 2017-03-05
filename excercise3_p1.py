'''
    File:        excercise2.py
    Description: Performs activities related to excercise 3:
                 Creates a neural networks between 2 and 10 neurons in the
                 hidden layer to test 3 pairs of datasets with 500, 1000 and 2000
                 vectors.
    Authors:     Joel Rivas        #11-10866
                 Leonardo Martinez #11-10576
                 Nicolas Manan     #06-39883
    Updated:     03/05/2017
'''
import numpy  as np
import pandas as pd
from neuralnetwork import *
from graphics      import draw_dataset_e31, draw_cost_curve

dictNominal={
        "Iris-virginica":0,
        "Iris-versicolor":0,
        "Iris-setosa":1
    }

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
        print("Min=%f,Max=%f"%(m,M))
        vector_min.append(m)
        vector_max.append(M)
        normalizedDataset[i]  = np.subtract(normalizedDataset[i], m)
        normalizedDataset[i]  = np.divide(normalizedDataset[i],M - m)

    return normalizedDataset, vector_min, vector_max

def readDataE3(trainset, normalize = False):

    dataset = pd.read_csv(trainset, delimiter = "," ,header = None,index_col = False)

    if normalize:
        #Normalize Dataset
        dataset, mean, std = normalize(dataset)
        #Change values of objective clases
        columnNominal=dataset[[4]]

        for row in columnNominal.values:
            row[0]=dictNominal[row[0]]

        dataset[[4]]=columnNominal

    #fix the dataset as an array of [x1, x2, x3,..., y]
    aux_dataset = list()
    for i in range(len(dataset)):
        row = dataset.iloc[i]
        aux_row = list()
        for j in range(len(row)):
            aux_row.append(row[j])

        aux_dataset.append(aux_row)

    dataset = aux_dataset

    return dataset

if __name__ == "__main__":

    #loading datasets for E3 part 1
    print("Starting...\nLoading data sets...")
    trainset50 = readData('data/Data_Exercise3/Datos_P3_50B.txt')
    trainset60 = readData('data/Data_Exercise3/Datos_P3_60B.txt')
    trainset70 = readData('data/Data_Exercise3/Datos_P3_70B.txt')
    trainset80 = readData('data/Data_Exercise3/Datos_P3_80B.txt')
    trainset90 = readData('data/Data_Exercise3/Datos_P3_90B.txt')

    print(trainset50)

    #drawing datasets
    # draw_dataset(trainset50)
    # draw_dataset(trainset60)
    # draw_dataset(trainset70)
    # draw_dataset(trainset80)
    # draw_dataset(trainset90)

    #load testsets
    testset50 = readData('data/Data_Exercise3/Datos_P3_50BR.txt')
    testset60 = readData('data/Data_Exercise3/Datos_P3_60BR.txt')
    testset70 = readData('data/Data_Exercise3/Datos_P3_70BR.txt')
    testset80 = readData('data/Data_Exercise3/Datos_P3_80BR.txt')
    testset90 = readData('data/Data_Exercise3/Datos_P3_90BR.txt')


    #drawing testset
    # draw_dataset(testset50)
    # draw_dataset(testset60)
    # draw_dataset(testset70)
    # draw_dataset(testset80)
    # draw_dataset(testset90)

    #using the same learnig rate alpha and epochs
    alpha  = 0.1
    epochs = 20000

    #for this problem, we are setting:
    #  - two neurons in the input layer (a point (x,y))
    #  - two neurons in the output layer, since we want to know whether the
    #    point belongs to the circle or the square.
    n_inputs   = 2
    n_outputs  = 2

    #Neurons range between 2 and 10
    neuron_range = [i for i in range(2,11)]

    #####################################################################
    #                          TRAINSET 1                               #
    #####################################################################

    datasetname = 'Trainset 1 N500'

    for neurons in neuron_range:
        network = init_network(n_inputs, neurons, n_outputs)

        print("Training %s with %d neurons, %d epochs and alpha = %f..."%(datasetname, neurons, epochs, alpha))
        iter_vs_cost = train(network, trainset1, alpha, epochs, n_outputs)
        print("Done.")

        # Draw the cost curve
        draw_cost_curve(datasetname, iter_vs_cost, alpha, neurons)

        print("Predicting...")
        predictedset, expected_vs_predicted = calculate_predictions(network, testset)
        print("Done.")

        total_error, false_positives, false_negatives = calculate_errors(expected_vs_predicted)

        draw_dataset(datasetname, predictedset, neurons, alpha, [total_error, false_positives, false_negatives])

    #####################################################################
    #                          TRAINSET 2                               #
    #####################################################################

    # datasetname = 'Trainset 2 N1000'
    #
    # for neurons in neuron_range:
    #     network = init_network(n_inputs, neurons, n_outputs)
    #
    #     print("Training %s with %d neurons, %d epochs and alpha = %f..."%(datasetname, neurons, epochs, alpha))
    #     iter_vs_cost = train(network, trainset2, alpha, epochs, n_outputs)
    #     print("Done.")
    #
    #     # Draw the cost curve
    #     draw_cost_curve(datasetname, iter_vs_cost, alpha, neurons)
    #
    #     print("Predicting...")
    #     predictedset, expected_vs_predicted = calculate_predictions(network, testset)
    #     print("Done.")
    #
    #     total_error, false_positives, false_negatives = calculate_errors(expected_vs_predicted)
    #
    #     draw_dataset(datasetname, predictedset, neurons, alpha, [total_error, false_positives, false_negatives])
    #
    # #####################################################################
    # #                          TRAINSET 3                               #
    # #####################################################################
    #
    # datasetname = 'Trainset 3 N2000'
    #
    # for neurons in neuron_range:
    #    network = init_network(n_inputs, neurons, n_outputs)
    #
    #    print("Training %s with %d neurons, %d epochs and alpha = %f..."%(datasetname, neurons, epochs, alpha))
    #    iter_vs_cost = train(network, trainset3, alpha, epochs, n_outputs)
    #    print("Done.")
    #
    #    # Draw the cost curve
    #    draw_cost_curve(datasetname, iter_vs_cost, alpha, neurons)
    #
    #    print("Predicting...")
    #    predictedset, expected_vs_predicted = calculate_predictions(network, testset)
    #    print("Done.")
    #
    #    total_error, false_positives, false_negatives = calculate_errors(expected_vs_predicted)
    #
    #    draw_dataset(datasetname, predictedset, neurons, alpha, [total_error, false_positives, false_negatives])
    #
    #
    # #####################################################################
    # #                          TRAINSET 4                               #
    # #####################################################################
    #
    # datasetname = 'Trainset 4 N500'
    #
    # for neurons in neuron_range:
    #    network = init_network(n_inputs, neurons, n_outputs)
    #
    #    print("Training %s with %d neurons, %d epochs and alpha = %f..."%(datasetname, neurons, epochs, alpha))
    #    iter_vs_cost = train(network, trainset4, alpha, epochs, n_outputs)
    #    print("Done.")
    #
    #    # Draw the cost curve
    #    draw_cost_curve(datasetname, iter_vs_cost, alpha, neurons)
    #
    #    print("Predicting...")
    #    predictedset, expected_vs_predicted = calculate_predictions(network, testset)
    #    print("Done.")
    #
    #    total_error, false_positives, false_negatives = calculate_errors(expected_vs_predicted)
    #
    #    draw_dataset(datasetname, predictedset, neurons, alpha, [total_error, false_positives, false_negatives])
    #
    #
    # #####################################################################
    # #                          TRAINSET 5                               #
    # #####################################################################
    #
    # datasetname = 'Trainset 5 N1000'
    #
    # for neurons in neuron_range:
    #     network = init_network(n_inputs, neurons, n_outputs)
    #
    #     print("Training %s with %d neurons, %d epochs and alpha = %f..."%(datasetname, neurons, epochs, alpha))
    #     iter_vs_cost = train(network, trainset5, alpha, epochs, n_outputs)
    #     print("Done.")
    #
    #     # Draw the cost curve
    #     draw_cost_curve(datasetname, iter_vs_cost, alpha, neurons)
    #
    #     print("Predicting...")
    #     predictedset, expected_vs_predicted = calculate_predictions(network, testset)
    #     print("Done.")
    #
    #     total_error, false_positives, false_negatives = calculate_errors(expected_vs_predicted)
    #
    #     draw_dataset(datasetname, predictedset, neurons, alpha, [total_error, false_positives, false_negatives])
    #
    # #####################################################################
    # #                          TRAINSET 6                               #
    # #####################################################################
    #
    # datasetname = 'Trainset 6 N1000'
    #
    # for neurons in neuron_range:
    #    network = init_network(n_inputs, neurons, n_outputs)
    #
    #    print("Training %s with %d neurons, %d epochs and alpha = %f..."%(datasetname, neurons, epochs, alpha))
    #    iter_vs_cost = train(network, trainset6, alpha, epochs, n_outputs)
    #    print("Done.")
    #
    #    # Draw the cost curve
    #    draw_cost_curve(datasetname, iter_vs_cost, alpha, neurons)
    #
    #    print("Predicting...")
    #    predictedset, expected_vs_predicted = calculate_predictions(network, testset)
    #    print("Done.")
    #
    #    total_error, false_positives, false_negatives = calculate_errors(expected_vs_predicted)
    #
    #    draw_dataset(datasetname, predictedset, neurons, alpha, [total_error, false_positives, false_negatives])


    try:
        input("Press enter to finish...")
    except SyntaxError:
        pass
