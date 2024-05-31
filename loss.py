import numpy as np

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class LossMSE(Loss):
    def forward(self, y_pred, y_true):
        return np.mean((y_true - y_pred) ** 2, axis=-1)

    def backward(self, deriv_values, y_true):
        samples = len(deriv_values)
        outputs = len(deriv_values[0])
        self.deriv_inputs = -2 * (y_true - deriv_values) / outputs
        self.deriv_inputs /= samples