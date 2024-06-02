import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_mse_training_loss(file_paths):
    plt.figure(figsize=(10, 6))
    for file_path, label in file_paths:
        data = pd.read_excel(file_path, sheet_name='training_losses')
        epochs = data['epoch']
        losses = data['loss']
        plt.plot(epochs, losses, label=label)

    plt.xlabel('Epoka')
    plt.ylabel('Błąd MSE')
    plt.yscale('log')
    plt.title('Błąd MSE na zbiorze uczącym')
    plt.legend()
    plt.grid(True)
    plt.savefig('charts/mse_training_loss_comparison.png')
    plt.show()


def plot_mse_test_loss(file_paths):
    plt.figure(figsize=(10, 6))
    for file_path, label in file_paths:
        data = pd.read_excel(file_path, sheet_name='test_losses')
        epochs = data['epoch']
        losses = data['loss']
        base_error = losses[0]
        plt.plot(epochs, losses, label=label)

    plt.axhline(y=base_error, color='r', linestyle='--', label='Błąd bazowy')
    plt.xlabel('Epoka')
    plt.ylabel('Błąd MSE')
    plt.yscale('log')
    plt.title('Błąd MSE na zbiorze testowym')
    plt.legend()
    plt.grid(True)
    plt.savefig('charts/mse_test_loss_comparison.png')
    plt.show()


def plot_error_distribution(result_data_paths):
    plt.figure(figsize=(10, 6))
    for file_path, label in result_data_paths:
        data = pd.read_excel(file_path)
        predicted_x = data['predictedX']
        predicted_y = data['predictedY']
        real_x = data['realX']
        real_y = data['realY']

        errors = np.sqrt((predicted_x - real_x) ** 2 + (predicted_y - real_y) ** 2)
        sorted_errors = np.sort(errors)
        cdf = np.arange(len(sorted_errors)) / float(len(sorted_errors))
        plt.plot(sorted_errors, cdf, label=label)

    plt.xlabel('Błąd')
    plt.ylabel('Dystrybuanta')
    plt.title('Dystrybuanta błędów lokalizacji')
    plt.legend()
    plt.grid(True)
    plt.savefig('charts/error_distribution_comparison.png')
    plt.show()


def plot_corrected_vs_actual(test_data_path, result_data_paths):
    test_data = pd.read_excel(test_data_path)
    x = test_data['measuredX']
    y = test_data['measuredY']
    ref_x = test_data['realX']
    ref_y = test_data['realY']

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label='Dane z czujników UWB', alpha=0.5)

    for file_path, label in result_data_paths:
        data = pd.read_excel(file_path)
        result_x = data['predictedX']
        result_y = data['predictedY']
        plt.plot(result_x, result_y, 'o', label=label, alpha=0.5)

    plt.plot(ref_x, ref_y, 'o', label='Dane referencyjne', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Działanie sieci neuronowej - dane testowe')
    plt.legend()
    plt.grid(True)
    plt.savefig('charts/corrected_vs_actual.png')
    plt.show()


if __name__ == "__main__":
    mse_train_files = [
        ('losses_1_layers_16.xlsx', '1 warstwa ukryta, 16 neuronów'),
        ('losses_2_layers_16_16.xlsx', '2 warstwy ukryte, 16 neuronów'),
        ('losses_3_layers_16_16_16.xlsx', '3 warstwy ukryte, 16 neuronów')
    ]

    mse_test_files = [
        ('losses_1_layers_16.xlsx', '1 warstwa ukryta, 16 neuronów'),
        ('losses_2_layers_16_16.xlsx', '2 warstwy ukryte, 16 neuronów'),
        ('losses_3_layers_16_16_16.xlsx', '3 warstwy ukryte, 16 neuronów')
    ]

    result_files = [
        ('result_data_1_layers_16.xlsx', '1 warstwa ukryta, 16 neuronów'),
        ('result_data_2_layers_16_16.xlsx', '2 warstwy ukryte, 16 neuronów'),
        ('result_data_3_layers_16_16_16.xlsx', '3 warstwy ukryte, 16 neuronów')
    ]

    plot_mse_training_loss(mse_train_files)
    plot_mse_test_loss(mse_test_files)
    plot_error_distribution(result_files)
    plot_corrected_vs_actual('test_data.xlsx', result_files)