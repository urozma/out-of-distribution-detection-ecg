from torch import nn
from copy import deepcopy
import torch.nn.functional as F
import torch


# class SimpleNet(nn.Module):
#     """Simple architecture to start with"""

#     def __init__(self, in_channels=1, **kwargs):
#         super().__init__()
#         # main part of the network
#         self.conv1 = nn.Conv1d(in_channels, 16, 7, padding='same')
#         self.conv2 = nn.Conv1d(16, 32, 5, padding='same')
#         self.conv3 = nn.Conv1d(32, 32, 5, padding='same')
#         self.conv4 = nn.Conv1d(32, 16, 5, padding='same')
#         self.conv5 = nn.Conv1d(16, 1, 3, padding='same')

#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = F.relu(self.conv2(out))
#         out = F.relu(self.conv3(out))
#         out = F.relu(self.conv4(out))
#         out = self.conv5(out)
#         return out

#     def get_copy(self):
#         """Get weights from the model"""
#         return deepcopy(self.state_dict())

#     def set_state_dict(self, state_dict):
#         """Load weights into the model"""
#         self.load_state_dict(deepcopy(state_dict))
#         return


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=6, **kwargs):
        super(UNet, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv1d(in_channels, 16, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.enc_conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.enc_conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        # Decoder
        self.dec_conv1 = nn.Conv1d(64, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.dec_conv2 = nn.Conv1d(64, 16, 5, padding=2)
        self.bn5 = nn.BatchNorm1d(16)
        self.dec_conv3 = nn.Conv1d(32, 16, 7, padding=3)
        self.bn6 = nn.BatchNorm1d(16)

        # Final output
        self.final_conv = nn.Conv1d(16, num_classes, 3, padding=1)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Encoder
        enc1 = F.relu(self.bn1(self.enc_conv1(x)))
        enc2 = F.relu(self.bn2(self.enc_conv2(F.max_pool1d(enc1, 2))))
        enc3 = F.relu(self.bn3(self.enc_conv3(F.max_pool1d(enc2, 2))))

        # Decoder
        dec1 = F.relu(self.bn4(self.dec_conv1(F.interpolate(enc3, scale_factor=2, mode='nearest'))))
        dec2 = F.relu(self.bn5(self.dec_conv2(F.interpolate(torch.cat([dec1, enc2], 1), scale_factor=2, mode='nearest'))))
        dec3 = F.relu(self.bn6(self.dec_conv3(torch.cat([dec2, enc1], 1))))

        # Dropout
        dec3 = self.dropout(dec3)

        # Final output
        out = self.final_conv(dec3)
        return out

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))


def simplenet(**kwargs):
    return UNet(**kwargs)