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

dictNominal={
        "Iris-virginica":2,
        "Iris-versicolor":1,
        "Iris-setosa":0
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

def calculate_errors_multiclas(results):
    '''
        Calculates errors for obtained results.
        Total MSE for Testing.
        # of False Positives
        # of False Negatives
    '''
    total_error     = 0
    c1_false_positives = 0
    c1_false_negatives = 0
    c2_false_positives = 0
    c2_false_negatives = 0
    c3_false_positives = 0
    c3_false_negatives = 0


    #this is the total MSQ for testing
    print(len(results))
    total_error += (1/len(results)) * sum([(result[0] - result[1])**2 for result in results])

    for result in results:
        #false positives for c1
        if result[0] == 1 and result[1] == 0:
            c1_false_positives += 1
        if result[0] == 2 and result[1] == 0:
            c1_false_positives += 1
        #false positives for c2
        if result[0] == 0 and result[1] == 1:
            c2_false_positives += 1
        if result[0] == 2 and result[1] == 1:
            c2_false_positives += 1
        #false positives for c3
        if result[0] == 0 and result[1] == 2:
            c3_false_positives += 1
        if result[0] == 1 and result[1] == 2:
            c3_false_positives += 1
        #false negatives for c1
        if result[0] == 0 and result[1] == 1:
            c1_false_negatives += 1
        if result[0] == 0 and result[1] == 2:
            c1_false_negatives += 1
        #false negatives for c2
        if result[0] == 1 and result[1] == 0:
            c2_false_negatives += 1
        if result[0] == 1 and result[1] == 2:
            c2_false_negatives += 1
        #false negatives for c3
        if result[0] == 2 and result[1] == 0:
            c3_false_negatives += 1
        if result[0] == 2 and result[1] == 1:
            c3_false_negatives += 1

    return total_error, c1_false_positives, c1_false_negatives, c2_false_positives, c2_false_negatives, c3_false_positives, c3_false_negatives


if __name__ == "__main__":

    #loading datasets for E3 part 1
    print("Starting...\nLoading data sets...")
    trainset50 = readDataE3('data/Data_Exercise3/Datos_P3_50.txt')
    trainset60 = readDataE3('data/Data_Exercise3/Datos_P3_60.txt')
    trainset70 = readDataE3('data/Data_Exercise3/Datos_P3_70.txt')
    trainset80 = readDataE3('data/Data_Exercise3/Datos_P3_80.txt')
    trainset90 = readDataE3('data/Data_Exercise3/Datos_P3_90.txt')

    #load testsets
    testset50 = readDataE3('data/Data_Exercise3/Datos_P3_50R.txt')
    testset60 = readDataE3('data/Data_Exercise3/Datos_P3_60R.txt')
    testset70 = readDataE3('data/Data_Exercise3/Datos_P3_70R.txt')
    testset80 = readDataE3('data/Data_Exercise3/Datos_P3_80R.txt')
    testset90 = readDataE3('data/Data_Exercise3/Datos_P3_90R.txt')

    #using the same learnig rate alpha and epochs
    alpha  = 0.1
    epochs = 500

    #for this problem, we are setting:
    #  - four neurons in the input layer (attributes of examples)
    #  - two neurons in the output layer, since we want to know whether the
    #    example belongs to the Iris Setosa or not.
    n_inputs   = 4
    n_outputs  = 3

    #Neurons range between 2 and 10
    neuron_range = [i for i in range(4,11)]

    #####################################################################
    #                          TRAINSET 50%                             #
    #####################################################################

    datasetname = 'Trainset 50%% Binary Classification'

    for neurons in neuron_range:
        network = init_network(n_inputs, neurons, n_outputs)

        print("Training %s with %d neurons, %d epochs and alpha = %f..."%(datasetname, neurons, epochs, alpha))
        iter_vs_cost = train(network, trainset50, alpha, epochs, n_outputs)
        print("Done.")

        print("Predicting using 50%% of the remaining data...")
        predictedset, expected_vs_predicted = calculate_predictions(network, testset50)
        print("Done.")


        total_error, c1_fp, c1_fn, c2_fp, c2_fn, c3_fp, c3_fn = calculate_errors_multiclas(expected_vs_predicted)

        # C1 "Iris-setosa"
        # C2 "Iris-versicolor"
        # C3 "Iris-virginica"

        print("---- RESULTS %s NEURONS = %d EPOCHS = %d ALPHA = %f----"%(datasetname, neurons, epochs, alpha))
        print("Max Training Cost: %f"%(np.max(iter_vs_cost[1])))
        print("Min Training Cost: %f"%(np.min(iter_vs_cost[1])))
        print("Testing Error: %f"%(total_error))
        print("False Positives Iris-Setosa: %d"%(c1_fp))
        print("False Negatives Iris-Setosa: %d"%(c1_fn))
        print("False Positives Iris-Versicolor: %d"%(c2_fp))
        print("False Negatives Iris-Versicolor: %d"%(c2_fn))
        print("False Positives Iris-Virginica: %d"%(c3_fp))
        print("False Negatives Iris-Virginica: %d"%(c3_fn))
        print("Total Error %% : %d"%(100*((c1_fp + c1_fn + c2_fp + c2_fn + c3_fp + c3_fn)/2)/len(testset50)))
        print("--------------------")

    #####################################################################
    #                          TRAINSET 60%                             #
    #####################################################################

    datasetname = 'Trainset 60%% Binary Classification'

    for neurons in neuron_range:
        network = init_network(n_inputs, neurons, n_outputs)

        print("Training %s with %d neurons, %d epochs and alpha = %f..."%(datasetname, neurons, epochs, alpha))
        iter_vs_cost = train(network, trainset60, alpha, epochs, n_outputs)
        print("Done.")

        print("Predicting using 60%% of the remaining data...")
        predictedset, expected_vs_predicted = calculate_predictions(network, testset60)
        print("Done.")


        total_error, c1_fp, c1_fn, c2_fp, c2_fn, c3_fp, c3_fn = calculate_errors_multiclas(expected_vs_predicted)

        # C1 "Iris-setosa"
        # C2 "Iris-versicolor"
        # C3 "Iris-virginica"

        print("---- RESULTS %s NEURONS = %d EPOCHS = %d ALPHA = %f----"%(datasetname, neurons, epochs, alpha))
        print("Max Training Cost: %f"%(np.max(iter_vs_cost[1])))
        print("Min Training Cost: %f"%(np.min(iter_vs_cost[1])))
        print("Testing Error: %f"%(total_error))
        print("False Positives Iris-Setosa: %d"%(c1_fp))
        print("False Negatives Iris-Setosa: %d"%(c1_fn))
        print("False Positives Iris-Versicolor: %d"%(c2_fp))
        print("False Negatives Iris-Versicolor: %d"%(c2_fn))
        print("False Positives Iris-Virginica: %d"%(c3_fp))
        print("False Negatives Iris-Virginica: %d"%(c3_fn))
        print("Total Error %% : %d"%(100*((c1_fp + c1_fn + c2_fp + c2_fn + c3_fp + c3_fn)/2)/len(testset60)))
        print("--------------------")

    #####################################################################
    #                          TRAINSET 70%                             #
    #####################################################################

    datasetname = 'Trainset 70%% Binary Classification'

    for neurons in neuron_range:
        network = init_network(n_inputs, neurons, n_outputs)

        print("Training %s with %d neurons, %d epochs and alpha = %f..."%(datasetname, neurons, epochs, alpha))
        iter_vs_cost = train(network, trainset70, alpha, epochs, n_outputs)
        print("Done.")

        print("Predicting using 70%% of the remaining data...")
        predictedset, expected_vs_predicted = calculate_predictions(network, testset70)
        print("Done.")


        total_error, c1_fp, c1_fn, c2_fp, c2_fn, c3_fp, c3_fn = calculate_errors_multiclas(expected_vs_predicted)

        # C1 "Iris-setosa"
        # C2 "Iris-versicolor"
        # C3 "Iris-virginica"

        print("---- RESULTS %s NEURONS = %d EPOCHS = %d ALPHA = %f----"%(datasetname, neurons, epochs, alpha))
        print("Max Training Cost: %f"%(np.max(iter_vs_cost[1])))
        print("Min Training Cost: %f"%(np.min(iter_vs_cost[1])))
        print("Testing Error: %f"%(total_error))
        print("False Positives Iris-Setosa: %d"%(c1_fp))
        print("False Negatives Iris-Setosa: %d"%(c1_fn))
        print("False Positives Iris-Versicolor: %d"%(c2_fp))
        print("False Negatives Iris-Versicolor: %d"%(c2_fn))
        print("False Positives Iris-Virginica: %d"%(c3_fp))
        print("False Negatives Iris-Virginica: %d"%(c3_fn))
        print("Total Error %% : %d"%(100*((c1_fp + c1_fn + c2_fp + c2_fn + c3_fp + c3_fn)/2)/len(testset70)))
        print("--------------------")

    #####################################################################
    #                          TRAINSET 80%                             #
    #####################################################################

    datasetname = 'Trainset 80%% Binary Classification'

    for neurons in neuron_range:
        network = init_network(n_inputs, neurons, n_outputs)

        print("Training %s with %d neurons, %d epochs and alpha = %f..."%(datasetname, neurons, epochs, alpha))
        iter_vs_cost = train(network, trainset80, alpha, epochs, n_outputs)
        print("Done.")

        print("Predicting using 80%% of the remaining data...")
        predictedset, expected_vs_predicted = calculate_predictions(network, testset80)
        print("Done.")


        total_error, c1_fp, c1_fn, c2_fp, c2_fn, c3_fp, c3_fn = calculate_errors_multiclas(expected_vs_predicted)

        # C1 "Iris-setosa"
        # C2 "Iris-versicolor"
        # C3 "Iris-virginica"

        print("---- RESULTS %s NEURONS = %d EPOCHS = %d ALPHA = %f----"%(datasetname, neurons, epochs, alpha))
        print("Max Training Cost: %f"%(np.max(iter_vs_cost[1])))
        print("Min Training Cost: %f"%(np.min(iter_vs_cost[1])))
        print("Testing Error: %f"%(total_error))
        print("False Positives Iris-Setosa: %d"%(c1_fp))
        print("False Negatives Iris-Setosa: %d"%(c1_fn))
        print("False Positives Iris-Versicolor: %d"%(c2_fp))
        print("False Negatives Iris-Versicolor: %d"%(c2_fn))
        print("False Positives Iris-Virginica: %d"%(c3_fp))
        print("False Negatives Iris-Virginica: %d"%(c3_fn))
        print("Total Error %% : %d"%(100*((c1_fp + c1_fn + c2_fp + c2_fn + c3_fp + c3_fn)/2)/len(testset80)))
        print("--------------------")

    #####################################################################
    #                          TRAINSET 90%                             #
    #####################################################################

    datasetname = 'Trainset 90%% Binary Classification'

    for neurons in neuron_range:
        network = init_network(n_inputs, neurons, n_outputs)

        print("Training %s with %d neurons, %d epochs and alpha = %f..."%(datasetname, neurons, epochs, alpha))
        iter_vs_cost = train(network, trainset90, alpha, epochs, n_outputs)
        print("Done.")

        print("Predicting using 90%% of the remaining data...")
        predictedset, expected_vs_predicted = calculate_predictions(network, testset90)
        print("Done.")


        total_error, c1_fp, c1_fn, c2_fp, c2_fn, c3_fp, c3_fn = calculate_errors_multiclas(expected_vs_predicted)

        # C1 "Iris-setosa"
        # C2 "Iris-versicolor"
        # C3 "Iris-virginica"

        print("---- RESULTS %s NEURONS = %d EPOCHS = %d ALPHA = %f----"%(datasetname, neurons, epochs, alpha))
        print("Max Training Cost: %f"%(np.max(iter_vs_cost[1])))
        print("Min Training Cost: %f"%(np.min(iter_vs_cost[1])))
        print("Testing Error: %f"%(total_error))
        print("False Positives Iris-Setosa: %d"%(c1_fp))
        print("False Negatives Iris-Setosa: %d"%(c1_fn))
        print("False Positives Iris-Versicolor: %d"%(c2_fp))
        print("False Negatives Iris-Versicolor: %d"%(c2_fn))
        print("False Positives Iris-Virginica: %d"%(c3_fp))
        print("False Negatives Iris-Virginica: %d"%(c3_fn))
        print("Total Error %% : %d"%(100*((c1_fp + c1_fn + c2_fp + c2_fn + c3_fp + c3_fn)/2)/len(testset90)))
        print("--------------------")

    try:
        input("Press enter to finish...")
    except SyntaxError:
        pass
