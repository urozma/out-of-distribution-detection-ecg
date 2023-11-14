import torch
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
    df_all = pd.read_csv('ludb_lead_ii_data_3.csv', header=None)
    df = df_all.loc[df_all.iloc[:, 3] == 'Sinus rhythm']
    print(len(df))
    # Assuming each signal has 5000 entries
    signal_length = 3000

    # Calculate the number of signals
    num_signals = df.shape[0] // signal_length

    # Extract the signal values and target values without column names
    signals = df.iloc[:num_signals * signal_length, 1].to_numpy().reshape(-1, signal_length)
    targets = df.iloc[:num_signals * signal_length, 2].to_numpy().reshape(-1, signal_length)

    trn_tuple, val_tuple, tst_tuple = split_dataset(signals, targets)

    # t, signal_matrix, target_matrix = dataset(1505)

    # trn_tuple, val_tuple, tst_tuple = split_dataset(signal_matrix, target_matrix, [1000, 500, 5])
    # initialize data structure
    all_data = {'trn': {'x': trn_tuple[0], 'y': trn_tuple[1]},
                'val': {'x': val_tuple[0], 'y': val_tuple[1]},
                'tst': {'x': tst_tuple[0], 'y': tst_tuple[1]}}

    return all_data

def get_loaders(batch_sz, num_work, pin_mem):
    """Apply transformations to Dataset and create the DataLoaders for each task"""

    # transformations
    trn_transform, tst_transform = get_transforms()

    # dataset
    all_data = get_data()
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

