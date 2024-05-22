import numpy as np
from neural_network import NeuralNetwork

def main():
    # Testowe dane
    x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0, 0], [1, 0], [1, 0], [0, 1]])

    nn = NeuralNetwork()
    nn.add_input_layer(2)  # Warstwa wejściowa z 2 neuronami
    nn.add_layer(4)  # Pierwsza warstwa ukryta - 4 neurony
    nn.add_layer(3)  # Druga warstwa ukryta - 3 neurony
    nn.add_layer(2)  # Warstwa wyjściowa z 2 neuronami

    # Trening
    print("\nTrening sieci neuronowej:")
    epochs = 1000
    learning_rate = 0.01
    nn.train(x_train, y_train, epochs, learning_rate)

    print("\n\nTest sieci neuronowej:")

    # Test sieci neuronowej
    print(nn.forward(np.array([0, 0])))
    print(nn.forward(np.array([0, 1])))
    print(nn.forward(np.array([1, 0])))
    print(nn.forward(np.array([1, 1])))


if __name__ == '__main__':
    main()
