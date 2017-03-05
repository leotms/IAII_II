import numpy as np

def create_trainset():

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
