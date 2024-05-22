import numpy as np

class Layer:
    def __init__(self, input_size, output_size, function, derivative):
        self.input_size = input_size
        self.output_size = output_size
        self.function = function
        self.derivative = derivative
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.random.randn(output_size) * 0.1
        self.last_input = None
        self.last_output = None

    def forward(self, input_data):
        self.last_input = input_data
        net_input = np.dot(input_data, self.weights) + self.biases
        self.last_output = self.function(net_input)
        return self.last_output

    def backward(self, output_error, learning_rate):
        delta = output_error * self.derivative(self.last_output)
        input_error = np.dot(delta, self.weights.T)
        weights_gradient = np.dot(self.last_input.T, delta)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * delta.mean(axis=0)
        return input_error
