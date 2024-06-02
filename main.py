import math

import numpy as np
import pandas as pd

from layer import Layer
from activations import ActivationReLU, ActivationLinear
from utils import split_to_batches
from neural_network import NeuralNetwork
from loss import LossMSE
from optimizer import OptimizerSGD

def load_data(file_path):
    # Wczytywanie danych z pliku Excel
    data = pd.read_excel(file_path)
    # Przemieszanie wierszy danych, aby uniknąć wpływu porządku danych
    data = data.sample(frac=1).reset_index(drop=True)
    return data

def prepare_data(data):
    # Podział danych na wejściowe i referencyjne
    X_data = data[['measuredX', 'measuredY']]
    y_data = data[['realX', 'realY']]

    # Przeskalowanie danych aby operować na małych liczbach
    X_data = (X_data + 500) / 10000
    y_data = (y_data + 500) / 10000

    return X_data, y_data

def train_and_evaluate(network, train_data_batches, train_ref_batches, test_data_batches, test_ref_batches, epochs=300):
    # Train the network
    train_losses = []
    test_losses = []
    for iteration in range(1, epochs + 1):
        current_loss = 0
        for data_batch, ref_batch in zip(train_data_batches, train_ref_batches):
            output = network.forward(data_batch)
            loss = network.loss.calculate(output, ref_batch)
            if not math.isnan(loss):
                current_loss += loss

            network.backward(output, ref_batch)
            network.update_params()

        train_losses.append(current_loss)

        if iteration % 10 == 0:
            print(f'Iteration: {iteration}   |   Loss: {current_loss:.5f}')
            test_loss = 0
            for data_batch, ref_batch in zip(test_data_batches, test_ref_batches):
                output = network.forward(data_batch)
                loss = network.loss.calculate(output, ref_batch)
                if not math.isnan(loss):
                    test_loss += loss
            test_losses.append(test_loss)

    # Evaluate the network
    test_result_loss = 0
    results = []
    refs = []
    for data_batch, ref_batch in zip(test_data_batches, test_ref_batches):
        output = network.forward(data_batch)
        results.append(output)
        refs.append(ref_batch)
        loss = network.loss.calculate(output, ref_batch)
        if not math.isnan(loss):
            test_result_loss += loss

    results_array = np.concatenate(results)
    refs_array = np.concatenate(refs)

    return results_array, refs_array, train_losses, test_losses, test_result_loss


def main(hidden_layers_configurations):
    np.random.seed(25)

    train_data = load_data('train_data.xlsx')
    X_train_data, y_train_data = prepare_data(train_data)

    test_data = load_data('test_data.xlsx')
    X_test_data, y_test_data = prepare_data(test_data)

    batch_size = 128
    train_data_batches = split_to_batches(X_train_data, batch_size)
    train_ref_batches = split_to_batches(y_train_data, batch_size)
    test_data_batches = split_to_batches(X_test_data, batch_size)
    test_ref_batches = split_to_batches(y_test_data, batch_size)

    all_results = []

    for config in hidden_layers_configurations:
        network = NeuralNetwork()
        input_size = config['input_size']
        for layer_size in config['layer_sizes']:
            network.add(Layer(input_size, layer_size))
            network.add(ActivationReLU())
            input_size = layer_size
        network.add(Layer(input_size, 2))
        network.add(ActivationLinear())
        network.set(loss=LossMSE(), optimizer=OptimizerSGD())

        results_array, refs_array, train_losses, test_losses, test_result_loss = train_and_evaluate(
            network, train_data_batches, train_ref_batches, test_data_batches, test_ref_batches
        )

        result_df = pd.DataFrame(results_array, columns=['predictedX', 'predictedY'])
        refs_df = pd.DataFrame(refs_array, columns=['realX', 'realY'])
        final_df = pd.concat([result_df, refs_df], axis=1)

        for column in final_df.columns:
            final_df[column] = (final_df[column] * 10000) - 500

        # Save results
        file_suffix = f"{len(config['layer_sizes'])}_layers_{'_'.join(map(str, config['layer_sizes']))}"
        final_df.to_excel(f"result_data_{file_suffix}.xlsx", index=False)

        train_loss_df = pd.DataFrame({
            'epoch': list(range(1, len(train_losses) + 1)),
            'loss': train_losses
        })
        test_loss_df = pd.DataFrame({
            'epoch': list(range(1, len(test_losses) * 10, 10)),
            'loss': test_losses
        })
        with pd.ExcelWriter(f'losses_{file_suffix}.xlsx') as writer:
            train_loss_df.to_excel(writer, sheet_name='training_losses', index=False)
            test_loss_df.to_excel(writer, sheet_name='test_losses', index=False)

        all_results.append((config, test_result_loss))

    best_config = min(all_results, key=lambda x: x[1])
    print(f"\nBest configuration: {best_config[0]}")
    print(f"Test loss: {best_config[1]}")

if __name__ == "__main__":
    hidden_layers_configurations = [
        {'layer_sizes': [16], 'input_size': 2},  # 1 warstwa ukryta, 16 neuronów
        {'layer_sizes': [16, 16], 'input_size': 2},  # 2 warstwy ukryte, 16 neuronów każda
        {'layer_sizes': [16, 16, 16], 'input_size': 2}  # 3 warstwy ukryte, 16 neuronów każda
    ]
    main(hidden_layers_configurations)