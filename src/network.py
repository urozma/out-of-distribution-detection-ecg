from torch import nn
from copy import deepcopy
import torch.nn.functional as F
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, padding=4):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class UNet(nn.Module):
    def __init__(self, data_type):
        super(UNet, self).__init__()
        if data_type == 'real':
            num_classes = 6
        elif data_type =='toy':
            num_classes = 6
        # Encoder
        self.enc_block1 = ResidualBlock(1, 8)
        self.enc_block2 = ResidualBlock(8, 16)
        self.enc_block3 = ResidualBlock(16, 32)
        self.enc_block4 = ResidualBlock(32, 64)

        # Decoder
        self.dec_block1 = ResidualBlock(64, 32)
        self.dec_block2 = ResidualBlock(64, 16)
        self.dec_block3 = ResidualBlock(32, 8)
        self.dec_block4 = ResidualBlock(16, 8)

        # Final output
        self.final_conv = nn.Conv1d(8, num_classes, 3, padding=1)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Encoder
        enc1 = self.enc_block1(x)
        enc2 = self.enc_block2(F.max_pool1d(enc1, 2))
        enc3 = self.enc_block3(F.max_pool1d(enc2, 2))
        enc4 = self.enc_block4(F.max_pool1d(enc3, 2))

        # Decoder
        dec1 = self.dec_block1(F.interpolate(enc4, scale_factor=2, mode='nearest'))
        dec2 = self.dec_block2(F.interpolate(torch.cat([dec1, enc3], 1), scale_factor=2, mode='nearest'))
        dec3 = self.dec_block3(F.interpolate(torch.cat([dec2, enc2], 1), scale_factor=2, mode='nearest'))
        dec4 = self.dec_block4(torch.cat([dec3, enc1], 1))

        # Dropout
        dec4 = self.dropout(dec4)

        # Final output
        out = self.final_conv(dec4)
        return out

    def extract_latent_space(self, x):
        # Encoder
        enc1 = self.enc_block1(x)
        enc2 = self.enc_block2(F.max_pool1d(enc1, 2))
        enc3 = self.enc_block3(F.max_pool1d(enc2, 2))
        enc4 = self.enc_block4(F.max_pool1d(enc3, 2))
        return enc4
    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))



def simplenet(**kwargs):
    return UNet(**kwargs)