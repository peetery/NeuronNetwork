import numpy as np

class ActivationReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, deriv_values):
        self.deriv_inputs = deriv_values.copy()
        self.deriv_inputs[self.inputs <= 0] = 0

class ActivationLinear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backward(self, deriv_values):
        self.deriv_inputs = deriv_values.copy()