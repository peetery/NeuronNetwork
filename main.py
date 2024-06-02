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

def main():
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

    network = NeuralNetwork()
    network.add(Layer(2, 16))  # Pierwsza warstwa ukryta
    network.add(ActivationReLU())
    network.add(Layer(16, 16))  # Druga warstwa ukryta
    network.add(ActivationReLU())
    network.add(Layer(16, 2))  # Warstwa wyjściowa
    network.add(ActivationLinear())
    network.set(loss=LossMSE(), optimizer=OptimizerSGD())

    starting_train_loss = 0
    for data_batch, ref_batch in zip(train_data_batches, train_ref_batches):
        output = network.forward(data_batch)
        loss = network.loss.calculate(output, ref_batch)
        if not math.isnan(loss):
            starting_train_loss += loss
    print(f"Starting train loss: {starting_train_loss}")

    starting_test_loss = 0
    for data_batch, ref_batch in zip(test_data_batches, test_ref_batches):
        output = network.forward(data_batch)
        loss = network.loss.calculate(output, ref_batch)
        if not math.isnan(loss):
            starting_test_loss += loss

    epochs = 300
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

    print(f"\nStarting test loss: {starting_test_loss}")
    print(f"Test loss after forward prop: {test_result_loss}")

    results_array = np.concatenate(results)
    refs_array = np.concatenate(refs)
    result_df = pd.DataFrame(results_array, columns=['predictedX', 'predictedY'])
    refs_df = pd.DataFrame(refs_array, columns=['realX', 'realY'])
    final_df = pd.concat([result_df, refs_df], axis=1)

    for column in final_df.columns:
        final_df[column] = (final_df[column] * 10000) - 500

    final_df.to_excel("result_data_hidden=X_verY.xlsx", index=False) # X - liczba warstw ukrytych, Y - wariant sieci
    print("Results saved to result_data.xlsx")

    train_loss_df = pd.DataFrame({
        'epoch': list(range(1, epochs + 1)),
        'loss': train_losses
    })
    test_loss_df = pd.DataFrame({
        'epoch': list(range(1, epochs + 1, 10)),
        'loss': test_losses
    })
    with pd.ExcelWriter('losses.xlsx') as writer:
        train_loss_df.to_excel(writer, sheet_name='training_losses', index=False)
        test_loss_df.to_excel(writer, sheet_name='test_losses', index=False)

if __name__ == '__main__':
    main()
