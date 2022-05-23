"""model_classes.py
Model classes for light curve classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """A block containing a convolution and all the fixings that go with it.
    Adapted from ramjet https://github.com/golmschenk/ramjet/blob/master/ramjet/models/components/light_curve_network_block.py
    N.B. removed spatial dropout.
    TODO port from keras
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pooling_size: int, dropout: float = 0.1,
                 batch_normalization: bool = True, non_linearity: str = "LeakyReLU"):
        super(ConvBlock, self).__init__()
        # self.convolution = nn.Conv1D(filters, kernel_size=kernel_size)
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
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
            # self.batch_normalization_input_reshape = Reshape([-1])
            # self.batch_normalization_output_reshape = Reshape([-1, filters])
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


class Ramjet(nn.Module):
    """1D CNN Architecture from 
    Identifying Planetary Transit Candidates in TESS Full-frame Image Light Curves via Convolutional Neural Networks, Olmschenk 2021
    https://iopscience.iop.org/article/10.3847/1538-3881/abf4c6
    """
    def __init__(self, bin_factor, output_dim=1, dropout=0.1):
        super(Ramjet, self).__init__()
        # bin factor determines the size of the model due to input dimension
        if bin_factor == 7:
            self.input_dim = 2700
        elif bin_factor == 3:
            self.input_dim = 2700*3
        else:
            raise ValueError("bin_factor must be 3 or 7")
    
        self.output_dim = output_dim
        self.dropout = dropout

        self.block0 = ConvBlock(in_channels=1, out_channels=8, kernel_size=3, pooling_size=2, batch_normalization=False,
                                             dropout=0)
        self.block1 = ConvBlock(in_channels=8, out_channels=8, kernel_size=3, pooling_size=2, dropout=self.dropout)
        self.block2 = ConvBlock(in_channels=8, out_channels=16, kernel_size=3, pooling_size=2, dropout=self.dropout)
        self.block3 = ConvBlock(in_channels=16, out_channels=32, kernel_size=3, pooling_size=2, dropout=self.dropout)
        self.block4 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, pooling_size=2, dropout=self.dropout)
        self.block5 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, pooling_size=2, dropout=self.dropout)
        self.block6 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, pooling_size=2, dropout=self.dropout) # another pool
        self.block7 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, pooling_size=1, dropout=self.dropout)
        
        self.block8 = DenseBlock(input_dim=128*17, output_dim=512, dropout=self.dropout)        # 16/17 is the number of features in the last conv block for 2600/2700 input.
        self.block9 = DenseBlock(input_dim=512, output_dim=20, dropout=0, batch_normalization=False)

        self.linear_out = nn.Linear(20, self.output_dim)


    def forward(self, x):
        # x: (B, LC_LEN)
        # LC_LEN = 2700 for binned sectors 10-14
        x = x.view(x.shape[0], 1, x.shape[-1])  # input shape: (B, 1, 2700)
        x = self.block0(x)              # (B, 8, 1349)
        x = self.block1(x)              # (B, 8, 673)
        x = self.block2(x)              # (B, 16, 335)
        x = self.block3(x)              # (B, 32, 166)
        x = self.block4(x)              # (B, 64, 82)
        x = self.block5(x)              # (B, 128, 40)
        x = self.block6(x)              # (B, 128, 19)
        x = self.block7(x)              # (B, 128, 17)
        x = x.view(x.shape[0], -1)      # (B, 128*17)
        x = self.block8(x)              # (B, 512)
        x = self.block9(x)              # (B, 20)
        outputs = self.linear_out(x)    # (B, 1)

        return outputs


class SimpleNetwork(nn.Module):
    """Generic fully connected MLP with adjustable depth.
    """
    def __init__(self, input_dim, hid_dims=[128, 64], output_dim=1, non_linearity="ReLU", dropout=0.0):
        """Params:
        - input_dim (int): input dimension
        - hid_dims (List[int]): list of hidden layer dimensions
        - output_dim (int): output dimension
        - non_linearity (str): type of non-linearity in hidden layers
        - dropout (float): dropout rate (applied each layer)
        """
        super(SimpleNetwork, self).__init__()

        dims = [input_dim]+hid_dims

        self.dropout = nn.Dropout(p=dropout)
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])
       
        self.act = get_activation(non_linearity)
        self.acts = nn.ModuleList([self.act for _ in range(len(dims)-1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, act in zip(self.fcs, self.acts):
            x = act(fc(self.dropout(x)))
        # non activated final layer
        return self.fc_out(x)



class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network
    """
    def __init__(self, input_dim, hidden_layer_dims=[128, 128, 64], output_dim=1, non_linearity="ReLU", dropout=0.0):
        """
        Parans:
        - input_dim (int): input dimension
        - hid_dims (List[int]): list of hidden layer dimensions
        - output_dim (int): output dimension
        - non_linearity (str): type of non-linearity in hidden layers
        - dropout (float): dropout rate (applied each layer)
        """
        super(ResidualNetwork, self).__init__()

        dims = [input_dim]+hidden_layer_dims

        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])

        self.res_fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False)
                                      if (dims[i] != dims[i+1])
                                      else nn.Identity()
                                      for i in range(len(dims)-1)])
        self.act = get_activation(non_linearity)
        self.acts = nn.ModuleList([self.act for _ in range(len(dims)-1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea))+res_fc(fea)

        return self.fc_out(fea)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


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
