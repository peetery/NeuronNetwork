import glob
import pandas as pd
import math

def load_train_data() -> pd.DataFrame:
    data_frames = []

    for building in ['f10', 'f8']:
        directory = f'data/{building}/stat/'
        file_list = glob.glob(directory + f'{building.lower()}_stat_*.csv')
        print(f'Loading training data from {directory}:')
        print(file_list)

        for file in file_list:
            print(f'Reading file: {file}')
            # Wczytujemy plik CSV bez nagłówków
            df = pd.read_csv(file, header=None)
            # Przypisujemy własne nazwy kolumn
            df.columns = ['measuredX', 'measuredY', 'realX', 'realY']
            data_frames.append(df)

    if not data_frames:
        raise ValueError("No training data files were found.")

    all_data = pd.concat(data_frames, axis=0, ignore_index=True)
    important_data = all_data[['measuredX', 'measuredY', 'realX', 'realY']].dropna()
    print('Training data loaded and combined successfully.')
    return important_data

def load_test_data() -> pd.DataFrame:
    data_frames = []

    for building in ['f10', 'f8']:
        directory = f'data/{building}/dyn/'
        file_list = glob.glob(directory + f'{building.lower()}_dyn_[1-3][pz].csv')
        print(f'Loading test data from {directory}:')
        print(file_list)

        for file in file_list:
            print(f'Reading file: {file}')
            # Wczytujemy plik CSV bez nagłówków
            df = pd.read_csv(file, header=None)
            # Przypisujemy własne nazwy kolumn
            df.columns = ['measuredX', 'measuredY', 'realX', 'realY']
            data_frames.append(df)

    if not data_frames:
        raise ValueError("No test data files were found.")

    all_data = pd.concat(data_frames, axis=0, ignore_index=True)
    important_data = all_data[['measuredX', 'measuredY', 'realX', 'realY']].dropna()
    print('Test data loaded and combined successfully.')
    return important_data

def split_to_batches(data, batch_size):
    if isinstance(data, pd.DataFrame):
        data = data.values
    number_of_data_points = len(data)
    nr_of_batches = math.ceil(number_of_data_points / batch_size)
    list_of_batches = []
    for i in range(nr_of_batches):
        batch = data[batch_size * i: batch_size * i + batch_size]
        list_of_batches.append(batch)
    return list_of_batches

if __name__ == '__main__':
    train_data = load_train_data()
    train_data.to_excel('train_data.xlsx', index=False)
    test_data = load_test_data()
    test_data.to_excel('test_data.xlsx', index=False)