import random as rd
import sys

def generate(dataset, quantity):
	
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
			




dataset = sys.argv[1]
quantity = int(sys.argv[2])

generate(dataset, quantity)
