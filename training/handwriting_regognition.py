""" First principles implementation of a neutral network to recognise handwritten digits.

This script provides a means to train and test a 3-layer neural netork. Training is done
using stochastic gradient descent with the MNIST dataset (available at  
http://yann.lecun.com/exdb/mnist/). Testing can be done with the MNIST test set or with
your own test set. See the required pre-processing steps at http://yann.lecun.com/exdb/mnist/
if you are using your own test set.
	
This script requires that the following libraries be installed  within the Python 
environment you are running this script in:

	numpy

 """

# ===================================== IMPORTS ======================================== #

import mnist_loader
import custom_fileIO
import sys
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})


# ================================= GLOBAL VARIABLES =================================== #

TRAINING_DATA, VALIDATION_DATA, TESTING_DATA_MNIST = mnist_loader.load_data_wrapper()
TESTING_DATA_WEB = custom_fileIO.readFromFile("dataset_testing5.json")
TESTING_DATA_SIZE_MNIST = 10000
TESTING_DATA_SIZE_WEB = np.shape(TESTING_DATA_WEB[0])[0]

TRAINING_DATA_SIZE = 10000
LEARNING_RATE = 1.5
TRAINING_ITERATIONS = 600
LAMBDA = 0

n_x = 784								# number of neurons in input layer
n_h = 300 								# number of neurons in hidden layer
n_y = 10								# number of neurons in output layer

np.random.seed(1)
W1 = np.random.randn(n_h,n_x)			# weights between input layer and hidden layer
b1 = np.zeros((n_h,1))					# biases between input layer and hidden layer
W2 = np.random.randn(n_y,n_h)			# weights between hidden layer and output layer
b2 = np.zeros((n_y,1))					# biases between hidden layer and output layer


# =============================== FUNCTION DEFINITIONS ================================= #

def sigmoid(x):
	"""Applies signmoid function to each element in input vector. """
	
	return 1/(1+np.exp(-x))


def softmax(x):
	"""Applies softmax function across all elements in input vector. """
	
	ans = []
	for i in x.T:
		exp = np.exp(i-np.max(i))
		result =  exp / exp.sum(axis=0)
		ans.append(result)
	
	ans = np.array(ans)
	
	return ans.T

def forward_prop(X):
	"""Performs forward propogation through the net. 
	
	Parameters
	----------
	X : numpy.ndarray
		2D numpy array containing all inputs
	
	Returns
	-------
	numpy.ndarray
		values of neurons in hidden layer A1
	numpy.ndarray
		values of neurons in output layer A2
	"""
	
	A1 = sigmoid(np.dot(W1, X) + b1) 
	A2 = softmax(np.dot(W2, A1) + b2) 
	
	return A1,A2


def calc_cost(A2, Y, m):
	"""Calculates the cost associated with current neural net parameters.
	
	The cost is defined as the sum of the cross entropy cost, and the L2 cost.  
	
	Parameters
	----------
	A2 : numpy.ndarray
		values of neurons in output layer
	Y : numpy.ndarray
		true value of outputs
	m : numpy.ndarray
		number of training inputs
	
	Returns
	-------
	number
		the cost associated with current neural net parameters.
	"""
	
	cross_entropy_cost = -np.sum(np.multiply(Y.T, np.log(A2))+ np.multiply(1-Y.T, np.log(1-A2)))/m
	L2_cost = (np.sum(np.square(W1)) + np.sum(np.square(W2)))*(LAMBDA/(2*m))
	
	return np.squeeze(cross_entropy_cost+L2_cost)


def back_prop(X, Y, A1, A2, m):
	"""Performs back progogation on the neural net using stocastic gradient decent.
	
	Parameters
	----------
	X : numpy.ndarray
		inputs to neural net
	Y : numpy.ndarray
		true value of outputs
	A1 : numpy.ndarray
		value of neurons in hidden layer
	A2 : numpy.ndarray
		value of neurons in output layer
	m : numpy.ndarray
		number of training inputs
	
	Returns
	-------
	number
		the change dW1 to apply to weights W1
	number
		the change db1 to apply to biases b1
	number
		the change dW2 to apply to weights W2
	number
		the change db2 to apply to biases W2
	"""
	
	
	dZ2 = A2 - Y.T
	dW2 = np.dot(dZ2, A1.T)/m  + (LAMBDA/m)*W2 
	db2 = np.sum(dZ2, axis=1, keepdims=True)/m
	dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
	dW1 = np.dot(dZ1, X.T)/m + (LAMBDA/m)*W1
	db1 = np.sum(dZ1, axis=1, keepdims=True)/m
	
	return dW1, db1, dW2, db2



def predict(X):
	"""Predict the current digit by passing input through the net.
	
	Parameters
	----------
	X : numpy.array
		array containing single input
	
	Returns
	-------
	numpy.array
		ouput array containing probabilities for each digit 0 to 9
	"""
	
	A1, A2 = forward_prop(X)
	y = np.squeeze(A2)
	y_predict = y.T
	return y_predict
	
	
def unvectorise_result(vect):
	"""Convert 10 element vector to single digit from 1 to 9 by using the index of the 
	highest value in the provided vector. 
	
	Parameters
	----------
	X : numpy.array
		array containing 10 elements each representing probability of digits 0 to 9.
	
	Returns
	-------
	number
		single digit produced from vector
	"""
	
	digitWithHighest = -1
	highestVal = -1
	for x in range(0, len(vect)):
		if(vect[x]>=highestVal):
			digitWithHighest = x
			highestVal = vect[x]
	return digitWithHighest



def train():
	""" Trains the model and updates the weights and biases uses in the net. Prints out
	the status of the net at regular intervals."""
	print("\n========================== TRAINING MODEL =============================\n")

	X = np.asarray(TRAINING_DATA[0][0:TRAINING_DATA_SIZE]).squeeze()
	X = X.transpose()
	Y = np.asarray(TRAINING_DATA[1][0:TRAINING_DATA_SIZE]).squeeze()
	
	m = X.shape[1]
	
	global W1, b1, W2, b2
	
	for i in range(0, TRAINING_ITERATIONS):
		A1, A2 = forward_prop(X)
		cost = calc_cost(A2, Y, m)
		dW1, db1, dW2, db2 = back_prop(X, Y, A1, A2, m)
	
		W1 = W1 - LEARNING_RATE*dW1
		b1 = b1 - LEARNING_RATE*db1
		W2 = W2 - LEARNING_RATE*dW2
		b2 = b2 - LEARNING_RATE*db2
		
		if(i%(100) == 0 or i==(TRAINING_ITERATIONS-1)):
			print("iteration={:>5d}  cost={:>10f}".format(i, cost), end=" ")
			printModelTest()



def printModelTest():
	"""Tests the model using both the MNIST test set and custom test set and prints 
	success rate of predictions to the console."""
	
	X1 = np.asarray(TESTING_DATA_MNIST[0][0:TESTING_DATA_SIZE_MNIST]).squeeze()
	X1 = X1.transpose()
	Y1 = np.asarray(TESTING_DATA_MNIST[1][0:TESTING_DATA_SIZE_MNIST]).squeeze()
	
	Y1_hat_bitmask = predict(X1)
	Y1_hat_digit = np.asarray([unvectorise_result(element) for element in Y1_hat_bitmask])
	error1 = Y1 - Y1_hat_digit
	bitmask1 = [element==0 for element in error1]
	accuracy1 = np.sum(bitmask1)/TESTING_DATA_SIZE_MNIST
	

	X2 = np.asarray(TESTING_DATA_WEB[0][0:TESTING_DATA_SIZE_WEB]).squeeze()
	X2 = X2.transpose()
	Y2 = np.asarray(TESTING_DATA_WEB[1][0:TESTING_DATA_SIZE_WEB]).squeeze()
	
	Y2_hat_bitmask = predict(X2)
	Y2_hat_digit = np.asarray([unvectorise_result(element) for element in Y2_hat_bitmask])
	error2 = Y2 - Y2_hat_digit
	bitmask2 = [element==0 for element in error2]
	accuracy2 = np.sum(bitmask2)/TESTING_DATA_SIZE_WEB

	
	print(" MNIST={:>3.3f}  WEB={:>3.3f}".format(accuracy1,accuracy2))


def test():
	"""Tests the model using both the MNIST test set and custom test set and prints 
	details of the test to the console."""
	
	print("\n===================== TESTING MODEL WITH MNIST ========================")
	
	X1 = np.asarray(TESTING_DATA_MNIST[0][0:TESTING_DATA_SIZE_MNIST]).squeeze()
	X1 = X1.transpose()
	Y1 = np.asarray(TESTING_DATA_MNIST[1][0:TESTING_DATA_SIZE_MNIST]).squeeze()
	
	Y1_hat_bitmask = predict(X1)
	Y1_hat_digit = np.asarray([unvectorise_result(element) for element in Y1_hat_bitmask])
	error1 = Y1 - Y1_hat_digit
	bitmask1 = [element==0 for element in error1]
	accuracy1 = np.sum(bitmask1)/TESTING_DATA_SIZE_MNIST
	
	np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
	print("\nPrediction:\ny=\n{}\ny_hat=\n{}".format(Y1, Y1_hat_digit))
	print("Accuracy=",accuracy1)
	print(Y1_hat_bitmask)
	
	print("\n====================== TESTING MODEL WITH WEB =========================")
	
	X2 = np.asarray(TESTING_DATA_WEB[0][0:TESTING_DATA_SIZE_WEB]).squeeze()
	X2 = X2.transpose()
	Y2 = np.asarray(TESTING_DATA_WEB[1][0:TESTING_DATA_SIZE_WEB]).squeeze()
	
	Y2_hat_bitmask = predict(X2)
	Y2_hat_digit = np.asarray([unvectorise_result(element) for element in Y2_hat_bitmask])
	error2 = Y2 - Y2_hat_digit
	bitmask2 = [element==0 for element in error2]
	accuracy2 = np.sum(bitmask2)/TESTING_DATA_SIZE_WEB
	
	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
	print("\nPrediction:\ny=\n{}\ny_hat=\n{}".format(Y2, Y2_hat_digit))
	print("Accuracy=", accuracy2)
	print(Y2_hat_bitmask)




def printElementFromDataset():
	"""Prints a single unnormalised input to the console - useful for debugging the web 
	interface."""
	
	np.set_printoptions(formatter={'float': lambda x: "{0:0.0f}".format(x)})
	vals = TESTING_DATA_WEB[0][35].squeeze()
	vals = vals * 255
	ans = np.array2string(vals, precision=0, separator=',', suppress_small=True) 
	print(ans)



if __name__ == "__main__":
	
	#W1, W2, b1, b2 = custom_fileIO.load_weights(n_x,n_h,n_y)
	train()
	test()
	custom_fileIO.save_weights(W1, W2, b1, b2)

	
	


# =================================== TESTING LOG ====================================== #

# MNIST
# Accuracy 79.1% -> training_size=1000, learning_rate=1.5, num_iterations=3000, n_hidden=100, sigmoid
# Accuracy 83.4% -> training_size=2000, learning_rate=1.5, num_iterations=3000, n_hidden=100, sigmoid
# Accuracy 88.2% -> training_size=10000, learning_rate=1.5, num_iterations=3000, n_hidden=100, sigmoid
# Accuracy 89.0% -> training_size=40000, learning_rate=1.5, num_iterations=3000, n_hidden=100, sigmoid
# Accuracy 89.0% -> training_size=10000, learning_rate=1.5, num_iterations=3000, n_hidden=100, softmax, lam=0.7
# Accuracy 89.8% -> training_size=10000, learning_rate=1.5, num_iterations=3000, n_hidden=100, softmax
# Accuracy 91.1% -> training_size=10000, learning_rate=1.5, num_iterations=3000, n_hidden=200, sigmoid
# Accuracy 91.4% -> training_size=10000, learning_rate=1.5, num_iterations=3000, n_hidden=200, softmax
# Accuracy 91.6% -> training_size=40000, learning_rate=1.5, num_iterations=10000, n_hidden=100, sigmoid
# Accuracy 92.0% -> training_size=50000, learning_rate=1.5, num_iterations=3000, n_hidden=100, softmax


# WEB 2
# Accuracy 64.0% (74.0%) -> training_size=10000, learning_rate=1.5, num_iterations=3000, n_hidden=200, sigmoid
# Accuracy 66.0% (75.0%) -> training_size=10000, learning_rate=1.5, num_iterations=3000, n_hidden=100, softmax
# Accuracy 68.0% -> training_size=10000, learning_rate=1.5, num_iterations=3000, n_hidden=100, sigmoid


# WEB 3 - line width 30px, bound image first then center
# Accuracy 80.0% (80.0%) -> training_size=10000, learning_rate=1.5, num_iterations=3000, n_hidden=100, softmax


# WEB 4 - N=100, line width 20px, (NB invalid inputs)
# Accuracy 75.0% (86.0%) -> training_size=10000, learning_rate=1.5, num_iterations=3000, n_hidden=100, softmax, lam=0.7
# Accuracy 82.0% (87.0%) -> training_size=10000, learning_rate=1.5, num_iterations=3000, n_hidden=100, softmax
# Accuracy 85.0% (87.0%) -> training_size=50000, learning_rate=1.5, num_iterations=3000, n_hidden=100, softmax


# WEB 5 - N=100, line width 20px 
# Accuracy 87.0% (95.0%) -> training_size=50000, learning_rate=1.5, num_iterations=3000, n_hidden=100, softmax
# Accuracy 90.0% (96.0%) -> training_size=10000, learning_rate=1.5, num_iterations=3000, n_hidden=100, softmax
# Accuracy 93.0% (99.0%) -> training_size=10000, learning_rate=1.5, num_iterations=3000, n_hidden=300, softmax
# Accuracy 94.0% (98.0%) -> training_size=10000, learning_rate=1.5, num_iterations=3000, n_hidden=200, softmax
# Accuracy 96.0% (96.0%) -> training_size=10000, learning_rate=1.5, num_iterations=200, n_hidden=100, softmax
# Accuracy 99.0% (99.0%) -> training_size=10000, learning_rate=1.5, num_iterations=600, n_hidden=300, softmax


