import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
import numpy as np

nnfs.init


class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

class Activation_ReLU:
     def forward(self,inputs):
         self.output = np.maximum(0,inputs)

class Activation_SoftMax:
    def forward(self,input):
        exp_values = np.exp(input-np.max(input,axis=1,keepdims=True))
        probabilities = exp_values/ np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self,output,y):
        sample_loss = self.foward(output,y)
        data_loss = np.mean(sample_loss)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def foward(self,y_pred,y_true):
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(sample),y_true]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped*y_true,axis=1)
        
        negative_log_likehoods = -np.log(correct_confidence)
        return negative_log_likehoods

# Create dataset
X, y = vertical_data(samples=100, classes=3)

# Create model
layer1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs
activation1 = Activation_ReLU()
layer2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs
activation2 = Activation_SoftMax()

# Create loss function
loss_function = Loss_CategoricalCrossEntropy()

# Helper variables
lowest_loss = 9999999  # some initial value
best_dense1_weights = layer1.weights.copy()
best_dense1_biases = layer1.biases.copy()
best_dense2_weights = layer2.weights.copy()
best_dense2_biases = layer2.biases.copy()

for iteration in range(10000):

    # Update weights with some small random values
    layer1.weights += 0.05 * np.random.randn(2, 3)
    layer1.biases += 0.05 * np.random.randn(1, 3)
    layer2.weights += 0.05 * np.random.randn(3, 3)
    layer2.biases += 0.05 * np.random.randn(1, 3)

    # Perform a forward pass of our training data through this layer
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    # Perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = loss_function.calculate(activation2.output, y)


    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions==y)

    # If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print('New set of weights found, iteration:', iteration,
              'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = layer1.weights.copy()
        best_dense1_biases = layer1.biases.copy()
        best_dense2_weights = layer2.weights.copy()
        best_dense2_biases = layer2.biases.copy()
        lowest_loss = loss
    # Revert weights and biases
    else:
        layer1.weights = best_dense1_weights.copy()
        layer1.biases = best_dense1_biases.copy()
        layer2.weights = best_dense2_weights.copy()
        layer2.biases = best_dense2_biases.copy()