# digit-recognition
Demonstration of how a neural network can be used to recognise handwritten digits. The neural network was built in Python from first principles and trained using stochastic gradient descent with the MNIST dataset. The neural net was ported to run in JavaScript in the browser. 

### Implementation notes
A three layer neural network is used with 784 input neurons, representing a 28x28 pixel image, fully connected to a hidden layer of 100 neurons. The output layer contains 10 fully connected neurons corresponding to each digit from 0 to 9. A softmax activation function is used in the output layer allowing each output to be interpreted as a probability. The model was trained using stochastic gradient descent with the MNIST dataset, available at http://yann.lecun.com/exdb/mnist/. As recommend for this dataset, each image is pre-processed before being passed through the net. This involves first scaling the image to fit in a 20x20 pixel bounding box, then centering the image on its center of mass, and finally normalising each pixel to have a value between 0 and 1. 
