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
    p1 = plt.scatter(inCircleX, inCircleY, c='r', marker='.', label = "Points inside circle.")
    p2 = plt.scatter(outCircleX, outCircleY, c='c', marker='.', label = "Points outside circle.")
    plt.legend(loc=2)

    if errors:
        total_values = len(dataset)
        total_error = errors[0]
        false_positives = errors[1]
        false_negatives = errors[2]

        info  = "Training Error: %f\nFalse Positives: %d  %.3f %% \nFalse Negatives: %d  %.3f %%  \n"%(total_error, false_positives, 100*false_positives/total_values, false_negatives, 100*false_negatives/total_values)

        plt.figtext(0.4, 0.8, info,
                bbox=dict(facecolor = 'white', alpha=0.5),
                horizontalalignment = 'left',
                verticalalignment   = 'center')

    plt.show(block=False)


def draw_cost_curve(trainset, iter_vs_cost, alpha, neurons):

    iterations = iter_vs_cost[0]
    cost       = iter_vs_cost[1]

    p1 = plt.plot(iterations,cost)
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
