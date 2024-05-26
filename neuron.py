import numpy as np
import random

class Neuron:
    def __init__(self, input_size, include_bias=True):
        self.weights = np.random.randn(input_size) * 0.5
        self.bias = random.uniform(-0.5, 0.5) if include_bias else 0.0
        self.include_bias = include_bias
        self.input_values = None
        self.net_sum = 0
        self.output = 0

    def calculate_net_sum(self):
        return np.dot(self.input_values, self.weights) + self.bias

    def calculate_output(self):
        self.net_sum = self.calculate_net_sum()
        self.output = 1 / (1 + np.exp(-self.net_sum))
        return self.output

    def calculate_output_derivative(self):
        return self.output * (1 - self.output)

    def update(self, input_values):
        self.input_values = np.array(input_values)
        self.calculate_output()
