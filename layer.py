import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, deriv_values):
        self.deriv_weights = np.dot(self.inputs.T, deriv_values)
        self.deriv_biases = np.sum(deriv_values, axis=0, keepdims=True)
        self.deriv_inputs = np.dot(deriv_values, self.weights.T)