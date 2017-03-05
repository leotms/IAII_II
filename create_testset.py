'''
    File:        create_testset.py
    Description: Creates a testset for excercise2 of a total of 10201 examples.
    Authors:     Joel Rivas        #11-10866
                 Nicolas Manan     #06-39883
                 Leonardo Martinez #11-10576
    Updated:     03/05/2017
'''

import numpy as np

def create_trainset():
    '''
         Creates a testset for excercise2 of a total of 10201 examples.
    '''

    points = np.arange(0, 20.2, 0.2)

    f = open('data/datos_P2_TESTSET.txt', 'w')

    for i in range(len(points)):
        for j in range(len(points)):
            value = ( (points[i]-10)**2 + (points[j]-10)**2 )
            if value <= 36:
                incircle = 1
            else:
                incircle = 0
            f.write("%f %f %d \n" % (points[i],points[j],incircle))

if __name__ == "__main__":
    create_trainset()
