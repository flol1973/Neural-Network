import nnfs
from nnfs.datasets import spiral_data
import numpy as np

nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def Derivative_Relu(self, input):
        input[input <= 0] = 0
        input[input > 0] = 1
        return input


class Activation_Sigmoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))

    def Derivative_Sigmoid(self, inputs):
        sigmoid = 1 / (1 + np.exp(-inputs))
        return sigmoid * (1 - sigmoid)


class Activation_SoftMax:
    def forward(self, input):
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_loss = self.forward(output, y)
        data_loss = np.mean(sample_loss)
        return data_loss


class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidence)
        return negative_log_likelihoods


class Loss_MeanSquare(Loss):
    def forward(self, y_pred, y_true):
        return (y_pred - y_true) ** 2

    def Derivative_MeanSquare(self, y_pred, y_true):
        y_true = y_true.reshape(-1, 1)  # Reshape y_true to match y_pred's shape
        y_true_one_hot = np.eye(y_pred.shape[1])[y_true].reshape(y_pred.shape)
        return 2 * (y_pred - y_true_one_hot) / y_pred.size


class Update_Weight:
    @staticmethod
    def update_weights(weights, gradient_weight, learning_rate):
        return weights - learning_rate * gradient_weight


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create model
layer1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs
activation1 = Activation_ReLU()
layer2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs
activation2 = Activation_Sigmoid()

# Create loss function
loss_function = Loss_MeanSquare()

# Forward pass
layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)

# Calculate gradient
derivative_MS = loss_function.Derivative_MeanSquare(activation2.output, y)
derivative_activation2 = activation2.Derivative_Sigmoid(layer2.output)
gradient_weight2 = np.dot(activation1.output.T, derivative_MS * derivative_activation2)

print("Gradient Weight 2:")
print(gradient_weight2)
