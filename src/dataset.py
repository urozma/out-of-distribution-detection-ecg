import torch
import pandas as pd
from torch.utils import data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from ECG_signal_generator import dataset
import numpy as np
from add_qs import add_qs_peaks_modified
from sklearn.utils import shuffle



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

    
def get_data(data_type, real_data_amount, health, add_qs):
    """Prepare data: dataset splits"""
    # Load the CSV file into a pandas DataFrame

    if data_type == 'real':
        df_all = pd.read_csv('ludb_data_w_fibrillation.csv', header=None)

        if isinstance(health, str):
            health = [health]
        df = df_all.loc[df_all.iloc[:, 3].isin(health)]

        # Assuming each signal has 5000 entries
        signal_length = 3000

        # Calculate the number of signals
        num_signals = df.shape[0] // signal_length

        # Extract the signal values and target values without column names
        signals = df.iloc[:num_signals * signal_length, 1].to_numpy().reshape(-1, signal_length)
        targets = df.iloc[:num_signals * signal_length, 2].to_numpy().reshape(-1, signal_length)

        # SHorten real data to desired length
        signals = signals[:real_data_amount]
        targets = targets[:real_data_amount]

        if add_qs is True:
            # Add QR
            targets = add_qs_peaks_modified(signals, targets)


    elif data_type == 'toy':
        # Get the toy dataset
        t, signals, targets = dataset(500, False)

    signals_norm = []
    for signal in signals:
        signals_norm.append(standarize(signal))
    trn_tuple, val_tuple, tst_tuple = split_dataset(signals_norm, targets)

    # Initialize data structure
    all_data = {'trn': {'x': trn_tuple[0], 'y': trn_tuple[1]},
                'val': {'x': val_tuple[0], 'y': val_tuple[1]},
                'tst': {'x': tst_tuple[0], 'y': tst_tuple[1]}}

    return all_data

def get_loaders(batch_sz, num_work, pin_mem, data_type, real_data_amount=136, health='Sinus rhythm', add_qs=False):
    """Apply transformations to Dataset and create the DataLoaders for each task"""

    # transformations
    trn_transform, tst_transform = get_transforms()

    # dataset
    all_data= get_data(data_type, real_data_amount, health, add_qs)
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
    np.random.seed(None)
    dataset, target = shuffle(dataset, target)

    tst_nr = 50
    trn_nr = round(0.7*len(dataset))
    val_nr = len(dataset)-tst_nr-trn_nr

    split = [tst_nr,val_nr,trn_nr]

    test_data = dataset[:split[0]]
    test_target = target[:split[0]]
    test_tuple = (test_data, test_target)

    val_data = dataset[split[0]:split[0] + split[1]]
    val_target = target[split[0]:split[0] + split[1]]
    val_tuple = (val_data, val_target)

    train_data = dataset[split[0] + split[1]:split[0] + split[1] + split[2]]
    train_target = target[split[0] + split[1]:split[0] + split[1] + split[2]]
    train_tuple = (train_data, train_target)

    return train_tuple, val_tuple, test_tuple

def normalize(signal, min_val=-1, max_val=1):
    # Normalize signal to range [-1, 1]
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    normalized_signal = (signal - signal_min) / (signal_max - signal_min)

    # Scale to desired range [min_val, max_val]
    scaled_signal = normalized_signal * (max_val - min_val) + min_val

    return scaled_signal

def standarize(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    standardized_data = (signal - mean) / std
    return standardized_data

def reverse_standarize(standardized_data, mean, std):
    original_data = standardized_data * std + mean
    return original_data
