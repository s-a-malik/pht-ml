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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, 
                 pooling_size: int, dropout: float = 0.1, batch_normalization: bool = True, non_linearity: str = "LeakyReLU"):
        super(ConvBlock, self).__init__()
        # self.convolution = nn.Conv1D(filters, kernel_size=kernel_size)
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.act = get_activation(non_linearity)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        if pooling_size > 1:
            self.max_pooling = nn.MaxPool1d(kernel_size=pooling_size)
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
            # if self.batch_normalization_input_reshape is not None: # TODO what is this for?
            #     x = self.batch_normalization_input_reshape(x)
            x = self.batch_normalization(x)
            # if self.batch_normalization_output_reshape is not None:
            #     x = self.batch_normalization_output_reshape(x)
        return x


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
    else:
        raise NameError(f"activation {name} not defined")
