import torch
import numpy as np
import pandas as pd
from torch.utils import data
from torch.utils.data import Dataset
import torchvision.transforms as transforms


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

    
def get_data():
    """Prepare data: dataset splits"""
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv('ludb_lead_ii_data_2.csv', header=None)

    # Assuming each signal has 5000 entries
    signal_length = 3000

    # Calculate the number of signals
    num_signals = df.shape[0] // signal_length

    # Extract the signal values and target values without column names
    signals = df.iloc[:num_signals * signal_length, 1].to_numpy().reshape(-1, signal_length)
    targets = df.iloc[:num_signals * signal_length, 2].to_numpy().reshape(-1, signal_length)

    trn_tuple, val_tuple, tst_tuple = split_dataset(signals, targets, [120, 55, 1])

    trn_tuple_seg, _, _ = segment_heartbeats(trn_tuple)
    val_tuple_seg, _, _ = segment_heartbeats(val_tuple)
    tst_tuple_seg, tst_start, original_lengths, = segment_heartbeats(tst_tuple)


    # t, signal_matrix, target_matrix = dataset(1505)

    # trn_tuple, val_tuple, tst_tuple = split_dataset(signal_matrix, target_matrix, [1000, 500, 5])
    #initialize data structure
    all_data = {'trn': {'x': trn_tuple_seg[0], 'y': trn_tuple_seg[1]},
                'val': {'x': val_tuple_seg[0], 'y': val_tuple_seg[1]},
                'tst': {'x': tst_tuple_seg[0], 'y': tst_tuple_seg[1]}}

    # all_data = {'trn': {'x': trn_tuple[0], 'y': trn_tuple[1]},
    #             'val': {'x': val_tuple[0], 'y': val_tuple[1]},
    #             'tst': {'x': tst_tuple[0], 'y': tst_tuple[1]}}

    return all_data, tst_start, original_lengths, 




def get_loaders(batch_sz, num_work, pin_mem):
    """Apply transformations to Dataset and create the DataLoaders for each task"""

    # transformations
    trn_transform, tst_transform = get_transforms()

    # dataset
    all_data, tst_start, original_lengths, = get_data()
    trn_dset = MemoryDataset(all_data['trn'], trn_transform)
    val_dset = MemoryDataset(all_data['val'], tst_transform)
    tst_dset = MemoryDataset(all_data['tst'], tst_transform)

    # loaders
    trn_load = data.DataLoader(trn_dset, batch_size=batch_sz, shuffle=True, num_workers=num_work, pin_memory=pin_mem)
    val_load = data.DataLoader(val_dset, batch_size=batch_sz, shuffle=False, num_workers=num_work, pin_memory=pin_mem)
    tst_load = data.DataLoader(tst_dset, batch_size=batch_sz, shuffle=False, num_workers=num_work, pin_memory=pin_mem)
    return trn_load, val_load, tst_load, tst_start, original_lengths


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
def split_dataset(dataset, target, split):
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


# def segment_heartbeats(data_tuple, window_length=300):
#     signals, targets = data_tuple
#     heartbeats_x = []
#     heartbeats_y = []
#     start_indices = []  # To save the starting index of each heartbeat segment

#     for signal, target in zip(signals, targets):
#         r_peaks_indices = np.where(target == 2)[0]

#         for index in r_peaks_indices:
#             start = index - window_length // 2
#             end = start + window_length

#             # # Correct the indices if they go out of bounds
#             # if start < 0:
#             #     start = 0
#             # if end > len(signal):
#             #     start = len(signal) - window_length
#             #     end = len(signal)

#             # Extract the heartbeat segment
#             heartbeat_signal = signal[start:end]
#             heartbeat_target = target[start:end]

#             if len(heartbeat_signal) < window_length:
#                 # Pad the heartbeat signal if it's shorter than the window length
#                 heartbeat_signal = np.pad(heartbeat_signal, (0, window_length - len(heartbeat_signal)), 'constant')
#                 heartbeat_target = np.pad(heartbeat_target, (0, window_length - len(heartbeat_target)), 'constant')

#             heartbeats_x.append(heartbeat_signal)
#             heartbeats_y.append(heartbeat_target)
#             start_indices.append(start)  # Save the start index

#     return (np.array(heartbeats_x), np.array(heartbeats_y)), np.array(start_indices)

def segment_heartbeats(data_tuple, window_length=300):
    signals, targets = data_tuple
    heartbeats_x = []
    heartbeats_y = []
    start_indices = []  # To save the starting index of each heartbeat segment
    original_lengths = []  # To save the length of the original segment before padding

    for signal, target in zip(signals, targets):
        r_peaks_indices = np.where(target == 2)[0]

        for index in r_peaks_indices:
            start = index - window_length // 2
            end = start + window_length

            if start < 0:
                start = 0
            if end > len(signal):
                end = len(signal)

            # Extract the heartbeat segment
            heartbeat_signal = signal[start:end]
            heartbeat_target = target[start:end]

            # Save the length of the original segment before any padding
            original_length = len(heartbeat_signal)

            if len(heartbeat_signal) < window_length:
                # Pad the heartbeat signal if it's shorter than the window length
                heartbeat_signal = np.pad(heartbeat_signal, (0, window_length - len(heartbeat_signal)), 'constant')
                heartbeat_target = np.pad(heartbeat_target, (0, window_length - len(heartbeat_target)), 'constant')

            heartbeats_x.append(heartbeat_signal)
            heartbeats_y.append(heartbeat_target)
            start_indices.append(start)
            original_lengths.append(original_length)  # Save the original length

    return (np.array(heartbeats_x), np.array(heartbeats_y)), np.array(start_indices), np.array(original_lengths)



# Example usage:
# Assuming 'signals_array' and 'targets_array' are your input 2D arrays:
# segmented_data, start_indices = segment_heartbeats((signals_array, targets_array))

