import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

np.random.seed(0)

'''X = [[1 ,2 ,3,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]]
'''

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def foward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

class Activation_ReLU:
     def foward(self,inputs):
         self.output = np.maximum(0,inputs)

class Activation_SoftMax:
    def foward(self,input):
        exp_values = np.exp(input-np.max(input,axis=1,keepdims=True))
        probabilities = exp_values/ np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilities

X, y = spiral_data(samples=100,classes=3)

layer1 = Layer_Dense(2,100)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(100,3)
activation2 = Activation_SoftMax()

layer1.foward(X)
activation1.foward(layer1.output)

layer2.foward(activation1.output)
activation2.foward(layer2.output)

print(activation2.output[:5])



