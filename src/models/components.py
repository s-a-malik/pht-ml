"""General Components of models
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """A block containing a convolution and all the fixings that go with it.
    Adapted from ramjet https://github.com/golmschenk/ramjet/blob/master/ramjet/models/components/light_curve_network_block.py
    N.B. removed spatial dropout.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding=1, 
                 pooling_size: int = 1, dropout: float = 0.1, batch_normalization: bool = True, non_linearity: str = "LeakyReLU"):
        super(ConvBlock, self).__init__()
        # self.convolution = nn.Conv1D(filters, kernel_size=kernel_size)
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.act = get_activation(non_linearity)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        if pooling_size > 1:
            # ceil mode true to get same output size as stride with kernel_size=1
            self.max_pooling = nn.MaxPool1d(kernel_size=pooling_size, ceil_mode=True)
        else:
            self.max_pooling = None
        if batch_normalization:
            self.batch_normalization = nn.BatchNorm1d(out_channels)
        else:
            self.batch_normalization = None

    def forward(self, x):
        """
        The forward pass of the layer.
        """
        x = self.conv(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.max_pooling is not None:
            x = self.max_pooling(x)
        if self.batch_normalization is not None:
            x = self.batch_normalization(x)
        return x


class ConvResBlock(nn.Module):
    """Two convolutional layers with a skip connection between them.
    Adapted from https://github.com/pytorch/vision/blob/a9a8220e0bcb4ce66a733f8c03a1c2f6c68d22cb/torchvision/models/resnet.py#L56-L72
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding=1, 
                 pooling_size: int = 1, dropout: float = 0.1, batch_normalization: bool = True, non_linearity: str = "LeakyReLU"):
        super(ConvResBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 pooling_size=pooling_size, dropout=dropout, batch_normalization=batch_normalization, non_linearity=non_linearity)
        if (stride > 1) or (pooling_size > 1) or (in_channels != out_channels):
            # TODO this is how resnet downsamples, could also pool instead
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride*pooling_size, padding=0),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = None
        self.conv2 = ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding,
                                 pooling_size=1, dropout=dropout, batch_normalization=batch_normalization, non_linearity=None)
        self.act2 = get_activation(non_linearity)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
            print("identity:", identity.shape)
        # skip connection
        out += identity
        out = self.act2(out)

        return out


class DenseBlock(nn.Module):
    """A block containing a dense layer and all the fixings that go with it.
    """

    def __init__(self, input_dim, output_dim: int = 1, dropout: float = 0.1,
                 batch_normalization: bool = True, non_linearity: str = "LeakyReLU"):
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.act = get_activation(non_linearity)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        if batch_normalization:
            self.batch_normalization = nn.BatchNorm1d(output_dim)
        else:
            self.batch_normalization = None

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.batch_normalization is not None:
            x = self.batch_normalization(x)
        return x


def get_activation(name):
    """Get activation function from name
    """
    if name == "ReLU":
        return nn.ReLU()
    elif name == "LeakyReLU":
        return nn.LeakyReLU(negative_slope=0.01)
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name is None:
        return nn.Identity()
    else:
        raise NameError(f"activation {name} not defined")
