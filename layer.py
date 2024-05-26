from neuron import Neuron
import numpy as np

class Layer:
    def __init__(self, num_neurons, input_size, include_bias=True):
        self.neurons = [Neuron(input_size, include_bias) for _ in range(num_neurons)]
        self.output = np.zeros(num_neurons)

    def forward(self, input_values):
        self.output = np.array([neuron.update(input_values) for neuron in self.neurons])
        return self.output

    def backward(self, output_error, learning_rate, momentum, prev_weight_grads, prev_bias_grads):
        weight_grads = []
        bias_grads = []
        for i, neuron in enumerate(self.neurons):
            error_signal = output_error[i] * neuron.calculate_output_derivative()
            input_error = error_signal * neuron.weights
            weight_grads.append(error_signal * neuron.input_values)
            bias_grads.append(error_signal)
            neuron.weights -= learning_rate * weight_grads[-1]
            if momentum and prev_weight_grads:
                neuron.weights -= momentum * learning_rate * prev_weight_grads[i]
            if neuron.include_bias:
                neuron.bias -= learning_rate * bias_grads[-1]
                if momentum and prev_bias_grads:
                    neuron.bias -= momentum * learning_rate * prev_bias_grads[i]
        return np.sum(input_error, axis=0), weight_grads, bias_grads