from layer import Layer
import functions
import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.input_size = None

    def add_input_layer(self, input_size):
        self.input_size = input_size

    def add_layer(self, output_size):
        input_size = self.layers[-1].output_size if self.layers else self.input_size
        if input_size is None:
            raise ValueError("Input size must be set before adding layers.")
        layer = Layer(input_size, output_size, functions.sigmoid, functions.sigmoid_derivative)
        self.layers.append(layer)

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, output_error, learning_rate):
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error, learning_rate)

    def train(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                output = self.forward(x)
                output_error = y - output
                self.backward(output_error, learning_rate)

            if epoch % 100 == 0:
                loss = np.mean(np.square(y_train - self.forward(x_train)))
                print(f"Epoch: {epoch}, Loss: {loss}")