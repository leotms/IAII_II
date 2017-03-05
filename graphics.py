'''
    File:        graphics.py
    Description: Provides functions for ploting data and analysis.
    Authors:     Leonardo Martinez #11-10576
                 Nicolas Manan     #06-39883
                 Joel Rivas        #11-10866
    Updated:     03/05/2017
'''

import numpy as np
import matplotlib.pyplot as plt

def split(dataset):
    inCircleX  = []
    inCircleY  = []
    outCircleX = []
    outCircleY = []

    for row in dataset:
        if row[-1] == 1:
            #point is inside circle
            inCircleX.append(row[0])
            inCircleY.append(row[1])
        else:
            outCircleX.append(row[0])
            outCircleY.append(row[1])

    return inCircleX, inCircleY, outCircleX, outCircleY


def draw_dataset(trainset, dataset, neurons, alpha, errors = None):

    inCircleX, inCircleY, outCircleX, outCircleY = split(dataset)

    #draw figure
    fig = plt.figure()
    ax = fig.add_subplot()

    plt.title('Testset results using : %s with %d Neurons and Alpha = %f'%(trainset,neurons,alpha))

    #plot points inside circle
    p1 = plt.scatter(inCircleX, inCircleY, c='#e74c3c', marker='.', label = "Inside circle.")
    p2 = plt.scatter(outCircleX, outCircleY, c='#1abc9c', marker='.', label = "Outside circle.")
    plt.legend(loc=3)

    if errors:
        total_values = len(dataset)
        total_error = errors[0]
        false_positives = errors[1]
        false_negatives = errors[2]

        info  = "Testing Error: %f\nFalse Positives: %d\nFalse Negatives: %d\n%% False Predictions: %.6f"%(total_error, false_positives, false_negatives, 100*(false_negatives+false_positives)/total_values)

        plt.figtext(0.4, 0.8, info,
                bbox=dict(facecolor = 'white', alpha=0.5),
                horizontalalignment = 'left',
                verticalalignment   = 'center')

    plt.show(block=False)


def draw_cost_curve(trainset, iter_vs_cost, alpha, neurons):

    iterations = iter_vs_cost[0]
    cost       = iter_vs_cost[1]

    fig = plt.figure()
    ax = fig.add_subplot()

    p1 = plt.plot(iterations,cost, c="#3498db")
    #Labels
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.title('Cost vs. Iterations\n%s with %d Neurons and Alpha = %f'%(trainset,neurons,alpha))

    #Calculate min and max cost
    mincost = np.min(cost)
    maxcost = np.max(cost)

    info  = "Min cost: " + str(mincost) +"\nMax cost: " + str(maxcost) + "\nAlpha: " + str(alpha)

    plt.figtext(0.4, 0.8, info,
            bbox=dict(facecolor = 'blue', alpha=0.2),
            horizontalalignment = 'left',
            verticalalignment   = 'center')
    plt.show(block=False)

def draw_cost_curves(trainset, all_iter_vs_cost, alpha, neurons):

    colors = ['#1abc9c', '#3498db', '#8e44ad', '#e74c3c', '#e67e22', '#f1c40f', '#34495e', '#7f8c8d', '#2ecc71']

    fig = plt.figure()
    ax = fig.add_subplot()

    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.title('Cost vs. Iterations\n%s with %d Neurons and Alpha = %f'%(trainset,neurons,alpha))

    for i in range(len(all_iter_vs_cost)):
        p = plt.plot(all_iter_vs_cost[i][0], all_iter_vs_cost[i][1], c=colors[i], marker='.', label = all_iter_vs_cost[i][2])

    plt.legend(loc=1)
    plt.show(block=False)
