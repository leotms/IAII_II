from neuralnetwork import *
from graphics      import draw_dataset, draw_cost_curve, draw_cost_curves

if __name__ == "__main__":

    #loading dataset
    besttrainset = readData('./data/datos_P2_EM2017_N500.txt')

    #drawing dataset
    # draw_dataset(besttrainset)

    #load testset
    testset =  readData('data/datos_P2_TESTSET.txt')

    # drawing testset
    #draw_dataset(testset)

    #using the same learnig rate alpha and epochs
    alpha  = 0.01
    epochs = 100

    #for this problem, we are setting:
    #  - two neurons in the input layer (a point (x,y))
    #  - two neurons in the output layer, since we want to know whether the
    #    point belongs to the circle or the square.
    n_inputs   = 2
    n_outputs  = 2

    #Neurons range between 2 and 10
    neuron_range = [i for i in range(2,11)]

    #Saving cost vs. iterations for grapic
    all_cost_vs_iterations = []

    #####################################################################
    #                          BEST DATASET 1                           #
    #####################################################################

    datasetname = 'Trainset 1 N500'

    for neurons in neuron_range:
        network = initialize(n_inputs, neurons, n_outputs)

        print("Training (BEST DATASET) %s with %d neurons, %d epochs and alpha = %f..."%(datasetname, neurons, epochs, alpha))
        iter_vs_cost = train(network, besttrainset, alpha, epochs, n_outputs)
        print("Done.")

        # save cost for later
        label = "%d Neurons"%(neurons)
        iter_vs_cost.append(label)
        all_cost_vs_iterations.append(iter_vs_cost)

        print("Predicting...")
        predictedset, expected_vs_predicted = calculate_predictions(network, testset)
        print("Done.")

        total_error, false_positives, false_negatives = calculate_errors(expected_vs_predicted)

        draw_dataset(datasetname, predictedset, neurons, alpha, [total_error, false_positives, false_negatives])

    #Print all cost vs functions for every neuros
    draw_cost_curves(datasetname, all_cost_vs_iterations, alpha, neurons)

    try:
        input("Press enter to finish...")
    except SyntaxError:
        pass
