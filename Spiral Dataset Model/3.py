import numpy as np

inputs = [1 ,2 ,3,2.5]

weights = [[-0.2,0.4,0.7,0.6],
           [0.2,-0.4,0.7,0.6],
           [0.22,0.4,-0.7,0.6]]

biases = [1,0.5,-0.7]

output = np.dot(weights,inputs) + biases

print(output)




'''
layers_outputs = []
for neuron_weight,neuron_bias in zip(weights,biases):
    neuron_output =0
    for n_input, weight in zip(input,neuron_weight):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layers_outputs.append(neuron_output)

print(layers_outputs)
'''