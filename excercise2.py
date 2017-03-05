from neuralnetwork import *
from graphics      import draw_dataset, draw_cost_curve

if __name__ == "__main__":

    #loading datasets
    trainset1 = readData('./data/datos_P2_EM2017_N500.txt')
    # trainset2 = readData('./data/datos_P2_EM2017_N1000.txt')
    # trainset3 = readData('./data/datos_P2_EM2017_N2000.txt')
    # trainset4 = readData('./data/datos_P2_EM2017_N500_2.txt')
    # trainset5 = readData('./data/datos_P2_EM2017_N1000_2.txt')
    # trainset6 = readData('./data/datos_P2_EM2017_N2000_2.txt')

    #drawing datasets
    # draw_dataset(trainset1)
    # draw_dataset(trainset2)
    # draw_dataset(trainset3)
    # draw_dataset(trainset4)
    # draw_dataset(trainset5)
    # draw_dataset(trainset6)

    #load testset
    testset =  readData('data/datos_P2_TESTSET.txt')

    #draw_dataset(testset)
    # drawing testset

    #using the same learnig rate alpha and epochs
    alpha  = 0.01
    epochs = 5000

    #for this problem, we are setting:
    #  - two neurons in the input layer (a point (x,y))
    #  - two neurons in the output layer, since we want to know whether the
    #    point belongs to the circle or the square.
    n_inputs   = 2
    n_outputs  = 2

    #Neurons range between 2 and 10
    neuron_range = [i for i in range(2,3)]

    #####################################################################
    #                          TRAINSET 1                               #
    #####################################################################

    datasetname = 'Trainset 1 N500'

    for neurons in neuron_range:
        network = initialize(n_inputs, neurons, n_outputs)

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
    #     network = initialize(n_inputs, neurons, n_outputs)
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

    #####################################################################
    #                          TRAINSET 3                               #
    #####################################################################

    # datasetname = 'Trainset 3 N2000'
    #
    # for neurons in neuron_range:
    #     network = initialize(n_inputs, neurons, n_outputs)
    #
    #     print("Training %s with %d neurons, %d epochs and alpha = %f..."%(datasetname, neurons, epochs, alpha))
    #     iter_vs_cost = train(network, trainset3, alpha, epochs, n_outputs)
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

    #####################################################################
    #                          TRAINSET 4                               #
    #####################################################################

    # datasetname = 'Trainset 4 N500'
    #
    # for neurons in neuron_range:
    #     network = initialize(n_inputs, neurons, n_outputs)
    #
    #     print("Training %s with %d neurons, %d epochs and alpha = %f..."%(datasetname, neurons, epochs, alpha))
    #     iter_vs_cost = train(network, trainset4, alpha, epochs, n_outputs)
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

    #####################################################################
    #                          TRAINSET 5                               #
    #####################################################################

    # datasetname = 'Trainset 5 N1000'
    #
    # for neurons in neuron_range:
    #     network = initialize(n_inputs, neurons, n_outputs)
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

    #####################################################################
    #                          TRAINSET 6                               #
    #####################################################################

    # datasetname = 'Trainset 6 N1000'
    #
    # for neurons in neuron_range:
    #     network = initialize(n_inputs, neurons, n_outputs)
    #
    #     print("Training %s with %d neurons, %d epochs and alpha = %f..."%(datasetname, neurons, epochs, alpha))
    #     iter_vs_cost = train(network, trainset6, alpha, epochs, n_outputs)
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

    try:
        input("Press enter to finish...")
    except SyntaxError:
        pass
