import torch
import pandas as pd
from torch.utils import data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from ECG_signal_generator import dataset
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.preprocessing import StandardScaler



class MemoryDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all images in memory"""

    def __init__(self, data, transform):
        """Initialization"""
        self.targets = [torch.FloatTensor(elem) for elem in data['y']]
        self.inputs = [torch.FloatTensor(elem) for elem in data['x']]
        self.transform = transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.inputs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = self.inputs[index]
        x = torch.unsqueeze(x, dim=0)
        # x = self.transform(x)
        y = self.targets[index]
        y = torch.unsqueeze(y, dim=0)
        
        return x, y

    
def get_data(data_type, real_data_amount):
    """Prepare data: dataset splits"""
    # Load the CSV file into a pandas DataFrame
    df_all = pd.read_csv('ludb_data.csv', header=None)
    df = df_all.loc[df_all.iloc[:, 3] == 'Sinus rhythm']

    # Assuming each signal has 5000 entries
    signal_length = 3000

    # Calculate the number of signals
    num_signals = df.shape[0] // signal_length

    # Extract the signal values and target values without column names
    signals_real = df.iloc[:num_signals * signal_length, 1].to_numpy().reshape(-1, signal_length)
    targets = df.iloc[:num_signals * signal_length, 2].to_numpy().reshape(-1, signal_length)

    # SHorten real data to desired length
    signals_real = signals_real[:real_data_amount]
    targets = targets[:real_data_amount]
    print(len(targets))


    # Get the toy dataset
    t, signals_toy, targets_toy = dataset(500, True)

    signals_real_norm = []
    signals_toy_norm = []
    for signal in signals_real:
        signals_real_norm.append(standarize(signal))
    for signal in signals_toy:
        signals_toy_norm.append(standarize(signal))

    #signals_real_norm, signals_toy_norm = signals_real, signals_toy

    if data_type == 'real':
        trn_tuple, val_tuple, tst_tuple = split_dataset(signals_real_norm, targets)

    elif data_type == 'toy':
        trn_tuple, val_tuple, tst_tuple = split_dataset(signals_toy_norm, targets_toy)

    elif data_type == 'combined':
        trn_tuple, val_tuple, _ = split_dataset(signals_toy_norm, targets_toy)
        _, _, tst_tuple = split_dataset(signals_real_norm, targets)

    plot_signal_histogram(signals_real_norm, title="Real data")
    plot_signal_histogram(signals_toy_norm[len(signals_real_norm):], title="Toy data")

    # initialize data structure
    all_data = {'trn': {'x': trn_tuple[0], 'y': trn_tuple[1]},
                'val': {'x': val_tuple[0], 'y': val_tuple[1]},
                'tst': {'x': tst_tuple[0], 'y': tst_tuple[1]}}

    return all_data

def get_loaders(batch_sz, num_work, pin_mem, data_type, real_data_amount):
    """Apply transformations to Dataset and create the DataLoaders for each task"""

    # transformations
    trn_transform, tst_transform = get_transforms()

    # dataset
    all_data = get_data(data_type, real_data_amount)
    trn_dset = MemoryDataset(all_data['trn'], trn_transform)
    val_dset = MemoryDataset(all_data['val'], tst_transform)
    tst_dset = MemoryDataset(all_data['tst'], tst_transform)

    # loaders
    trn_load = data.DataLoader(trn_dset, batch_size=batch_sz, shuffle=True, num_workers=num_work, pin_memory=pin_mem)
    val_load = data.DataLoader(val_dset, batch_size=batch_sz, shuffle=False, num_workers=num_work, pin_memory=pin_mem)
    tst_load = data.DataLoader(tst_dset, batch_size=batch_sz, shuffle=False, num_workers=num_work, pin_memory=pin_mem)
    return trn_load, val_load, tst_load


def get_transforms():
    """Unpack transformations and apply to train or test splits"""

    trn_transform_list = []
    tst_transform_list = []

    # padding
    # trn_transform_list.append(transforms.Pad(0))
    # tst_transform_list.append(transforms.Pad(0))

    # to tensor
    trn_transform_list.append(transforms.ToTensor())
    tst_transform_list.append(transforms.ToTensor())

    return transforms.Compose(trn_transform_list), transforms.Compose(tst_transform_list)

# Split dataset into pairs of training, validation and test data with their target values
def split_dataset(dataset, target):

    tst_nr = 5
    trn_nr = round(0.7*len(dataset))
    val_nr = len(dataset)-tst_nr-trn_nr

    split = [trn_nr,val_nr,tst_nr]

    train_data = dataset[:split[0]]
    train_target = target[:split[0]]
    train_tuple = (train_data, train_target)

    val_data = dataset[split[0]:split[0] + split[1]]
    val_target = target[split[0]:split[0] + split[1]]
    val_tuple = (val_data, val_target)

    test_data = dataset[split[0] + split[1]:split[0] + split[1] + split[2]]
    test_target = target[split[0] + split[1]:split[0] + split[1] + split[2]]
    test_tuple = (test_data, test_target)

    return train_tuple, val_tuple, test_tuple

def plot_signal_histogram(signals, title="Signal Histogram"):
    # Flatten the list of signals
    flattened_signals = list(itertools.chain(*signals))

    # Count the total number of signal values
    total_signals = len(flattened_signals)

    # Create histogram with specified range (-1 to 1) and get the counts
    counts, bins = np.histogram(flattened_signals, bins=40, range=(-1, 1))

    # Calculate the percentages
    percentages = (counts / total_signals) * 100

    # Plotting the histogram
    plt.bar(bins[:-1], percentages, width=np.diff(bins), align="edge")

    # Setting the x-axis limits
    plt.xlim(-2, 5)
    plt.ylim(0, 10)

    # Adding titles and labels
    plt.title(title)
    plt.xlabel("Signal value")
    plt.ylabel("Percentage")

    # Show plot
    plt.show()

def normalize(signal, min_val=-1, max_val=1):
    # Normalize signal to range [-1, 1]
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    normalized_signal = (signal - signal_min) / (signal_max - signal_min)

    # Scale to desired range [min_val, max_val]
    scaled_signal = normalized_signal * (max_val - min_val) + min_val

    return scaled_signal

def standardize_datasets(toy_dataset, real_dataset):
    scaler = StandardScaler()

    # Fit the scaler on the toy dataset and transform both datasets
    scaler.fit(toy_dataset)
    toy_dataset_standardized = scaler.transform(toy_dataset)
    real_dataset_standardized = scaler.transform(real_dataset)

    return toy_dataset_standardized, real_dataset_standardized

def standarize(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    standardized_data = (signal - mean) / std
    return standardized_data