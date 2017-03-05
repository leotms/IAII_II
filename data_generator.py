'''
    File:        data_generator.py
    Description: Generates trainsets for the excercise2 given the number of examples desired.
    Authors:     Joel Rivas        #11-10866
                 Leonardo Martinez #11-10576
                 Nicolas Manan     #06-39883
    Updated:     03/05/2017
'''

import random as rd
import sys

def generate(dataset, quantity):

	'''
		Generates examples equal in number to quantity and stores them in the file
		dataset.
	'''

	file = open(dataset, 'w')
	count = 1

	circle = 0
	inf_circle = int(quantity*0.4)
	sup_circle = quantity//2

	inf_nocircle=int(quantity*0.5)
	nocircle=0
	puntos=[]

	while count <= quantity:
		a = rd.uniform(0,20)
		b = rd.uniform(0,20)

		if a != b and not((a,b) in puntos):
			if (((a-10)**2 + (b-10)**2)<= 36):
				if (not(inf_circle < circle < sup_circle)):
					file.write(str(a) + " " +str(b) + " " + str(1) + " \n")
					circle += 1
					count += 1
					puntos.append((a,b))
				continue

			if (not(inf_nocircle < nocircle < sup_circle)):
				file.write(str(a) + " " +str(b) + " " + str(0) + " \n")
				count += 1
				puntos.append((a,b))

	file.close()

if __name__ == "__main__":

	dataset = sys.argv[1]
	quantity = int(sys.argv[2])

	generate(dataset, quantity)
