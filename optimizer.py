import numpy as np

from layer import Layer


class OptimizerSGD:
    def __init__(self, learning_rate=0.05):
        self.learning_rate = learning_rate

    def update_params(self, layer: Layer):
        if not hasattr(layer, 'weights'):
            return
        layer.weights += -self.learning_rate * layer.deriv_weights
        layer.biases += -self.learning_rate * layer.deriv_biases
