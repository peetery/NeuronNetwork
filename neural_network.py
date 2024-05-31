class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, X):
        inputs = X
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.output
        return inputs

    def backward(self, output, y):
        self.loss.backward(output, y)
        deriv_values = self.loss.deriv_inputs

        for layer in reversed(self.layers):
            layer.backward(deriv_values)
            deriv_values = layer.deriv_inputs

    def update_params(self):
        for layer in self.layers:
            self.optimizer.update_params(layer)