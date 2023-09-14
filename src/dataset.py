from torch.utils import data
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MemoryDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all images in memory"""

    def __init__(self, data, transform):
        """Initialization"""
        self.targets = data['y']
        self.inputs = data['x']
        self.transform = transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.inputs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = self.inputs[index]
        x = self.transform(x)
        y = self.targets[index]
        return x, y


def get_data():
    """Prepare data: dataset splits"""

    # initialize data structure
    all_data = {'trn': {'x': [], 'y': []}, 'val': {'x': [], 'y': []}, 'tst': {'x': [], 'y': []}}

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
