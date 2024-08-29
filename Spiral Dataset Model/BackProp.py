import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
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

     def Derivative_Relu(self,input):
         input[input<=0] = 0
         input[input>0] = 1
         return input
         
class Activation_sigmoid:
    def forward(self,inputs):
        self.output = 1/(1+np.exp(inputs))
    def Derivative_Sigmoid(self,inputs):
        input = np.array(inputs)
        return input*(1-input)
    
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

class Loss_meanSquare(Loss):
    def foward(self,y_pred,y_true):
        sample = len(y_pred)
        y_pred_clipped = np.array(np.clip(y_pred,1e-7,1-1e-7))
        mean_square = (y_pred_clipped-np.array(y_true))**2
        return mean_square
    def Derivative_MeanSquare(self,y_pred,y_true):
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(sample),y_true]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped*y_true,axis=1)
        return 2*(correct_confidence-1)

class Update_Weight:
    def Updation_Weigths(weights, gradient_Weight,Learning_rate):
        return weights - Learning_rate*gradient_Weight
    
# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create model
layer1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs
activation1 = Activation_ReLU()
layer2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs
activation2 = Activation_sigmoid()

# Create loss function
loss_function = Loss_meanSquare()

# Helper variables
lowest_loss = 9999999  # some initial value
best_dense1_weights = layer1.weights.copy()
best_dense1_biases = layer1.biases.copy()
best_dense2_weights = layer2.weights.copy()
best_dense2_biases = layer2.biases.copy()

layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)

derivative_MS = Loss_meanSquare()
gradient_weight2 =  np.dot(np.dot(derivative_MS.Derivative_MeanSquare(activation2.output,y),(activation2.Derivative_Sigmoid(layer2.output)).T),activation1.output)

print(gradient_weight2)