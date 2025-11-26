"""
Shuffle-BiLSTM Model Architecture

Components:
1. Shuffle Units (Group Conv + Channel Shuffle + Residual)
2. BiLSTM layers for temporal feature extraction
3. Classification head (FC + SoftMax)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupConv2d(nn.Module):
    """Group Convolution Layer"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, groups=1):
        super(GroupConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding, groups=groups, bias=False)
        
    def forward(self, x):
        return self.conv(x)


class ChannelShuffle(nn.Module):
    """Channel Shuffle Operation"""
    
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        
        # Reshape: (batch, channels, H, W) -> (batch, groups, channels_per_group, H, W)
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        
        # Transpose: (batch, groups, channels_per_group, H, W) -> (batch, channels_per_group, groups, H, W)
        x = x.transpose(1, 2).contiguous()
        
        # Flatten: (batch, channels_per_group, groups, H, W) -> (batch, channels, H, W)
        x = x.view(batch_size, channels, height, width)
        
        return x


class ShuffleUnit(nn.Module):
    """
    Shuffle Unit: GroupConv -> BN -> LeakyReLU -> ChannelShuffle -> GroupConv -> BN
    With residual connection
    """
    
    def __init__(self, in_channels, out_channels, groups=4, stride=1):
        super(ShuffleUnit, self).__init__()
        
        self.stride = stride
        self.groups = groups
        
        # First group convolution (1x1)
        self.gconv1 = GroupConv2d(in_channels, out_channels, kernel_size=1, 
                                  groups=groups)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=True)
        
        # Channel shuffle
        self.shuffle = ChannelShuffle(groups)
        
        # Depthwise convolution (3x3)
        self.dwconv = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, groups=out_channels, 
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Second group convolution (1x1)
        self.gconv2 = GroupConv2d(out_channels, out_channels, kernel_size=1,
                                  groups=groups)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        
        # First group conv
        out = self.leaky_relu(self.bn1(self.gconv1(x)))
        
        # Channel shuffle
        out = self.shuffle(out)
        
        # Depthwise conv
        out = self.leaky_relu(self.bn2(self.dwconv(out)))
        
        # Second group conv
        out = self.bn3(self.gconv2(out))
        
        # Residual addition
        out = self.leaky_relu(out + shortcut)
        
        return out


class ShuffleBiLSTM(nn.Module):
    """
    Complete Shuffle-BiLSTM Network for Boring Bar Vibration Monitoring
    
    Architecture:
    Input (256x256x3) -> Initial Conv -> Shuffle Units -> BiLSTM -> FC -> Output (3 classes)
    """
    
    def __init__(self, config):
        super(ShuffleBiLSTM, self).__init__()
        
        self.num_classes = config['data']['num_classes']
        self.num_groups = config['model']['num_groups']
        self.num_shuffle_units = config['model']['shuffle_units']
        self.bilstm_hidden = config['model']['bilstm_hidden']
        self.bilstm_layers = config['model']['bilstm_layers']
        self.dropout_rate = config['model']['dropout']
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Shuffle units
        self.shuffle_units = nn.ModuleList()
        in_channels = 64
        out_channels = 128
        
        for i in range(self.num_shuffle_units):
            stride = 2 if i == 0 else 1
            self.shuffle_units.append(
                ShuffleUnit(in_channels, out_channels, groups=self.num_groups, stride=stride)
            )
            in_channels = out_channels
            if i < self.num_shuffle_units - 1:
                out_channels *= 2
        
        # Global average pooling to get fixed-size feature vector
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # BiLSTM layers
        self.bilstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=self.bilstm_hidden,
            num_layers=self.bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate if self.bilstm_layers > 1 else 0
        )
        
        # Fully connected classification head
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.bilstm_hidden * 2, 256),  # *2 for bidirectional
            nn.LeakyReLU(0.01),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, self.num_classes)
        )
    
    def forward(self, x):
        # Input: (batch, 3, 256, 256)
        
        # Initial convolution
        out = self.init_conv(x)  # (batch, 64, 64, 64)
        
        # Shuffle units
        for shuffle_unit in self.shuffle_units:
            out = shuffle_unit(out)  # (batch, 512, 8, 8) after all units
        
        # Global pooling
        out = self.global_pool(out)  # (batch, 512, 1, 1)
        out = out.view(out.size(0), out.size(1))  # (batch, 512)
        
        # Expand for BiLSTM: treat as sequence of length 1
        out = out.unsqueeze(1)  # (batch, 1, 512)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(out)  # (batch, 1, 256)
        lstm_out = lstm_out[:, -1, :]  # Take last time step (batch, 256)
        
        # Classification
        logits = self.fc(lstm_out)  # (batch, 3)
        
        return logits


def test_model():
    """Test model architecture"""
    import yaml
    
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = ShuffleBiLSTM(config)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 256, 256)
    output = model(dummy_input)
    
    print(f"Model output shape: {output.shape}")
    print(f"Expected: ({batch_size}, {config['data']['num_classes']})")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Target from paper: ~1.9M parameters")


if __name__ == '__main__':
    test_model()
