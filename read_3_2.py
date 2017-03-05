import numpy  as np
import pandas as pd
from random import random, seed
from math   import exp

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
        print("Min=%f,Max=%f"%(m,M))
        vector_min.append(m)
        vector_max.append(M)
        normalizedDataset[i]  = np.subtract(normalizedDataset[i], m)
        normalizedDataset[i]  = np.divide(normalizedDataset[i],M - m)

    return normalizedDataset, vector_min, vector_max

def readData(trainset):

    dataset = pd.read_csv(trainset, delimiter = "," ,header = None,index_col = False)
    dataset, mean, std = normalize(dataset)

    columnNominal=dataset[[4]]

    for row in columnNominal.values:
        row[0]=dictNominal[row[0]]

    dataset[[4]]=columnNominal
    return dataset

if __name__=="__main__":

    resultData=readData("Irirs_dataset.txt")

    destfilepath = './CleanDatasetProcessed.csv'
    resultData.to_csv(destfilepath , index=False)




