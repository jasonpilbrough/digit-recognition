import mnist_loader
import custom_fileIO
import sys
import numpy as np
#np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

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


# GLOBAL VARIABLES
TRAINING_DATA, VALIDATION_DATA, TESTING_DATA_MNIST = mnist_loader.load_data_wrapper()
TESTING_DATA_WEB = custom_fileIO.readFromFile("dataset_testing5.json")


TESTING_DATA_SIZE_MNIST = 10000
TESTING_DATA_SIZE_WEB = np.shape(TESTING_DATA_WEB[0])[0]


TRAINING_DATA_SIZE = 10000
LEARNING_RATE = 1.5 #0.3 3 5 1.5
TRAINING_ITERATIONS = 600
LAMBDA = 0

n_x = 784
n_h = 300 #30 50 100
n_y = 10

np.random.seed(1)
W1 = np.random.randn(n_h,n_x)
b1 = np.zeros((n_h,1))
W2 = np.random.randn(n_y,n_h)
b2 = np.zeros((n_y,1))


#X = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
#X = X.transpose()
#Y = np.array([[1,0], [1,1], [0,1], [1,1], [0,1], [1,1], [1,1], [1,0]])

#n_x = 3
#n_hidden = 30
#n_y = 2


#print("Starting weights:\nW1={}\nb1={}\nW2={}\nb2={}".format(W1, b1, W2, b2))
#print("\nEnding weights:\nW1={}\nb1={}\nW2={}\nb2={}".format(W1, b1, W2, b2))


def sigmoid(x):
	return 1/(1+np.exp(-x))


def softmax(x):
	ans = []
	for i in x.T:
		exp = np.exp(i-np.max(i))
		result =  exp / exp.sum(axis=0)
		ans.append(result)
	
	ans = np.array(ans)
	
	return ans.T

def forward_prop(X):
	A1 = sigmoid(np.dot(W1, X) + b1) 
	#A2 = sigmoid(np.dot(W2, A1) + b2)
	A2 = softmax(np.dot(W2, A1) + b2) 
	
	return A1,A2


def calc_cost(A2, Y, m):
	cross_entropy_cost = -np.sum(np.multiply(Y.T, np.log(A2))+ np.multiply(1-Y.T, np.log(1-A2)))/m
	L2_cost = (np.sum(np.square(W1)) + np.sum(np.square(W2)))*(LAMBDA/(2*m))
	
	return np.squeeze(cross_entropy_cost+L2_cost)


def back_prop(X, Y, A1, A2, m):
	dZ2 = A2 - Y.T
	dW2 = np.dot(dZ2, A1.T)/m  + (LAMBDA/m)*W2 
	db2 = np.sum(dZ2, axis=1, keepdims=True)/m
	dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
	dW1 = np.dot(dZ1, X.T)/m + (LAMBDA/m)*W1
	db1 = np.sum(dZ1, axis=1, keepdims=True)/m
	
	return dW1, db1, dW2, db2



def predict(X):
	A1, A2 = forward_prop(X)
	y = np.squeeze(A2)
	y_predict = y.T
	return y_predict
	
	
def unvectorise_result(vect):
	digitWithHighest = -1
	highestVal = -1
	for x in range(0, len(vect)):
		if(vect[x]>=highestVal):
			digitWithHighest = x
			highestVal = vect[x]
	return digitWithHighest



def train():

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
	
	#printElementFromDataset()
	
	
























"""
def fire(x,w):
	return 1/(1+np.exp(-np.dot(x,w)))
	



training_inputs = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
training_outputs = np.array([0,0,0,1,0,0,0,1]).T
#training_outputs = np.array([0,0,0,0,1,1,1,1]).T


#training_inputs = np.array([[0,0,1], [1,1,1],[1,0,1], [0,1,1]])
#training_outputs = np.array([0,1,1,0]).T

np.random.seed(1)
w = 2 * np.random.random((3,3)) - 1

print("Starting weights w:", w)

for i in range(0,20000):
	y = fire(training_inputs,w)
	delta_w = (training_outputs-y) * y * (1 - y)
	w = w + np.dot(training_inputs.T, delta_w)


print("Ending weights w:", w)

print("Outputs: ", y)

x = [0,0,1]
y = fire(x,w)
print("x=", x," y=",y)
"""

"""

	
def web_train():


	#X = np.asarray(TRAINING_DATA[0][0:TRAINING_DATA_SIZE]).squeeze()
	#X = X.transpose()
	#Y = np.asarray(TRAINING_DATA[1][0:TRAINING_DATA_SIZE]).squeeze()
	
	
	X,Y = readFromFile("dataset_training.json")
	X = X.transpose()
	
	print("Number of training values: ",np.shape(X)[1])
	
	Ycopy = []
	for i in range(0, len(Y)):
		res =  mnist_loader.vectorized_result(Y[i])
		Ycopy.append(res.squeeze())
	
	Y = np.asarray(Ycopy)
	
	
	m = X.shape[1]
	
	#np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
	#print(TRAINING_DATA[0][1].squeeze())
	#X = np.round(X/255+0.2) * 255
	#X = np.ceil(X)
	
	global W1, b1, W2, b2
	
	for i in range(0, TRAINING_ITERATIONS):
		A1, A2 = forward_prop(X)
		cost = calc_cost(A2, Y, m)
		dW1, db1, dW2, db2 = back_prop(X, Y, A1, A2, m)
	
		W1 = W1 - LEARNING_RATE*dW1
		b1 = b1 - LEARNING_RATE*db1
		W2 = W2 - LEARNING_RATE*dW2
		b2 = b2 - LEARNING_RATE*db2
		
		if(i%(1000) == 0):
			print('Cost after iteration# {:d}: {:f}'.format(i, cost))
"""


