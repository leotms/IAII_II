from neuralnetwork import *
from graphics      import draw_dataset, draw_cost_curve

if __name__ == "__main__":

    #loading datasets
    trainset1 = readData('./data/datos_P2_EM2017_N500.txt')
    trainset2 = readData('./data/datos_P2_EM2017_N1000.txt')
    trainset3 = readData('./data/datos_P2_EM2017_N2000.txt')

    #drawing datasets
    # draw_dataset(trainset1)
    # draw_dataset(trainset2)
    # draw_dataset(trainset3)

    #load testset
    testset =  readData('./data/datos_P2_TRAINSET.txt')

    #draw_dataset(testset)
    # drawing testset

    #using the same learnig rate alpha and epochs
    alpha  = 0.3
    epochs = 20000

    #for this problem, we are setting:
    #  - two neurons in the input layer (a point (x,y))
    #  - two neurons in the output layer, since we want to know whether the
    #    point belongs to the circle or the square.
    n_inputs   = 2
    n_outputs  = 2


    # Initializing network for Trainset 1 N500. Neurons = 2.
    neurons = 2
    network = initialize(n_inputs, neurons, n_outputs)

    print("Training...")
    iter_vs_cost = train(network, trainset1, alpha, epochs, n_outputs)
    print("Done.")

    # Draw the cost curve
    draw_cost_curve("Trainset 1 N500", iter_vs_cost, alpha, neurons)

    print("Predicting...")
    predictedset = calculate_predictions(network, testset)
    print("Done.")
    
    draw_dataset(predictedset)
