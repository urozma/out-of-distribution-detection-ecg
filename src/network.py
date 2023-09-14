from torch import nn
from copy import deepcopy
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """Simple architecture to start with"""

    def __init__(self, in_channels=1, **kwargs):
        super().__init__()
        # main part of the network
        self.conv1 = nn.Conv1d(in_channels, 16, 7)
        self.conv2 = nn.Conv1d(16, 32, 5)
        self.conv3 = nn.Conv1d(32, 32, 5)
        self.conv4 = nn.Conv1d(32, 16, 5)
        self.lastconv = nn.Conv1d(16, 1, 3)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.lastconv(out))
        return out

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

def simplenet(**kwargs):
    return SimpleNet(**kwargs)
