import numpy as np

np.random.seed(0)

X = [[1 ,2 ,3,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]]

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def foward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.foward(X)
'''print(layer1.output)'''

layer2.foward(layer1.output)
print(layer2.output)
























'''
weights = [[-0.2,0.4,0.7,0.6],
           [0.2,-0.4,0.7,0.6],
           [0.22,0.4,-0.7,0.6]]

biases = [1,0.5,-0.7]

weights2 = [[-0.1,0.35,0.45],
           [0.33,-0.26,0.39],
           [0.32,0.12,-0.76]]

biases2 = [1.2,2,-0.5]


layer1_output = np.dot(inputs,np.array(weights).T ) + biases
layer2_output = np.dot(layer1_output,np.array(weights2).T ) + biases2

print(layer2_output)
'''