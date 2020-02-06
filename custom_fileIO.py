""" A library to load the custom image data and save/load net weights and biases. 

This script requires that the following libraries be installed within the Python 
environment you are running this script in:

	numpy
	

"""

# ===================================== IMPORTS ======================================== #


import numpy as np
import json


# =============================== FUNCTION DEFINITIONS ================================= #

def readFromFile(filename):
	"""Reads a custom test dataset from file and returns a set of labeled, normalised 
	inputs.
	
	Parameters
	----------
	filename : string
		name of dataset file to read - file must be in same dir as this script
	
	Returns
	-------
	numpy.ndarray
		2D array X containing all inputs 
	numpy.array
		1D array Y containing labeled outputs 
	"""

	with open(filename) as f:
		data = json.load(f)
	X = np.asarray(data["x"])
	X = X/255 #normalise input pixels to have a value between 0 and 1
	Y = np.asarray(data["y"])
	Y = Y.astype(int)
	
	return X,Y




def save_weights(W1, W2, b1, b2):
	"""Saves net weights and biases in appropriately named files. 
	
	Parameters
	----------
	W1 : numpy.ndarray
		weights between neurons in input layer and hidden layer
	W2 : numpy.ndarray
		weights between neurons in hidden layer and output layer
	b1 : numpy.ndarray
		biases between neurons in input layer and hidden layer
	b2 : numpy.ndarray
		biases between neurons in hidden layer and output layer
	
	"""
	
	f=open('net_weights_W1.txt','wb')
	np.savetxt(f,W1, delimiter=',',fmt="%0.3f", newline="],\n[")	
	f.close()
	f=open('net_weights_b1.txt','wb')
	np.savetxt(f,b1, delimiter=',',fmt="%0.3f", newline=",\n")	
	f.close()
	f=open('net_weights_W2.txt','wb')
	np.savetxt(f,W2, delimiter=',',fmt="%0.3f", newline="],\n[")	
	f.close()
	f=open('net_weights_b2.txt','wb')
	np.savetxt(f,b2, delimiter=',',fmt="%0.3f", newline=",\n")	
	f.close()



def load_weights(n_x,n_h,n_y):
	"""Reads in net weights and biases from file. 
	
	Parameters
	----------
	n_x : int
		number of neurons in input layer
	n_h : int
		number of neurons in hidden layer
	n_y : int
		number of neurons in output layer
		
	Returns
	-------
	numpy.ndarray
		weights between neurons in input layer and hidden layer	
	numpy.ndarray
		weights between neurons in hidden layer and output layer
	numpy.ndarray
		biases between neurons in input layer and hidden layer
	numpy.ndarray
		biases between neurons in hidden layer and output layer
	
	"""
	
	tempW1 = []
	with open('net_weights_W1.txt','r') as f: 
		contents = f.readlines()
		for x in contents:
			temparr = x.replace("],","").replace("[","").replace("\n","").replace("'","").split(",")
			for y in temparr:
				#check for empty string
				if not y:
					continue
				tempW1.append(float(y))	
	W1 = np.array(tempW1).reshape(n_h,n_x)
	
	tempW2 = []
	with open('net_weights_W2.txt','r') as f: 
		contents = f.readlines()
		for x in contents:
			temparr = x.replace("],","").replace("[","").replace("\n","").replace("'","").split(",")
			for y in temparr:
				#check for empty string
				if not y:
					continue
				tempW2.append(float(y))
			
	W2 = np.array(tempW2).reshape(n_y,n_h)
	
	tempb1 = []
	with open('net_weights_b1.txt','r') as f: 
		contents = f.readlines()
		for x in contents:
			tempb1.append(float(x.replace(",\n","")))
	b1 = np.expand_dims(np.array(tempb1), axis=1)
	
	tempb2 = []
	with open('net_weights_b2.txt','r') as f: 
		contents = f.readlines()
		for x in contents:
			tempb2.append(float(x.replace(",\n","")))
	b2 = np.expand_dims(np.array(tempb2), axis=1)
	
	return W1, W2, b1, b2

