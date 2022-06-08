"""model_classes.py
Model classes for light curve classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components import ConvBlock, ConvResBlock, DenseBlock, get_activation


class RamjetBin3(nn.Module):
    """1D CNN Architecture from 
    Identifying Planetary Transit Candidates in TESS Full-frame Image Light Curves via Convolutional Neural Networks, Olmschenk 2021
    https://iopscience.iop.org/article/10.3847/1538-3881/abf4c6
    """
    def __init__(self, input_dim=6300, output_dim=1, dropout=0.1):
        super(RamjetBin3, self).__init__()
    
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.block0 = ConvBlock(in_channels=1, out_channels=8, kernel_size=3, pooling_size=2, padding=0, stride=1,
                                batch_normalization=False, dropout=0)
        self.block1 = ConvBlock(in_channels=8, out_channels=8, kernel_size=3, pooling_size=2, dropout=self.dropout)
        self.block2 = ConvBlock(in_channels=8, out_channels=16, kernel_size=3, pooling_size=2, dropout=self.dropout)
        self.block3 = ConvBlock(in_channels=16, out_channels=32, kernel_size=3, pooling_size=2, dropout=self.dropout)
        self.block4 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, pooling_size=2, dropout=self.dropout)
        self.block5 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, pooling_size=2, dropout=self.dropout)
        self.block6 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, pooling_size=2, dropout=self.dropout)
        self.block7 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pooling_size=2, dropout=self.dropout)
        self.block8 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pooling_size=2, dropout=self.dropout)
        self.block9 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pooling_size=1, dropout=self.dropout)

        if self.input_dim == 6300:
            self.block10 = DenseBlock(input_dim=256*8, output_dim=512, dropout=self.dropout)
        elif self.input_dim == 5833:
            self.block10 = DenseBlock(input_dim=256*7, output_dim=512, dropout=self.dropout)
        else:
            raise ValueError('input_dim not supported')
        self.block11 = DenseBlock(input_dim=512, output_dim=20, dropout=0, batch_normalization=False)

        self.linear_out = nn.Linear(20, self.output_dim)


    def forward(self, x):
        # x: (B, LC_LEN)
        # LC_LEN = 6300 for binned sectors 10-14
        x = x.view(x.shape[0], 1, x.shape[-1])  # input shape: (B, 1, 6300)
        x = self.block0(x)              # (B, 8, 3149)
        x = self.block1(x)              # (B, 8, 1573)
        x = self.block2(x)              # (B, 16, 785)
        x = self.block3(x)              # (B, 32, 391)
        x = self.block4(x)              # (B, 64, 194)
        x = self.block5(x)              # (B, 128, 96)
        x = self.block6(x)              # (B, 256, 47)
        x = self.block7(x)              # (B, 256, 22)
        x = self.block8(x)              # (B, 256, 10)
        x = self.block9(x)              # (B, 256, 8)
        x = x.view(x.shape[0], -1)      # (B, 256*8)
        x = self.block10(x)              # (B, 512)
        x = self.block11(x)              # (B, 20)
        outputs = self.linear_out(x)    # (B, 1)

        return outputs


class RamjetBin7(nn.Module):
    """1D CNN Architecture from 
    Identifying Planetary Transit Candidates in TESS Full-frame Image Light Curves via Convolutional Neural Networks, Olmschenk 2021
    https://iopscience.iop.org/article/10.3847/1538-3881/abf4c6
    """
    def __init__(self, input_dim=2700, output_dim=1, dropout=0.1):
        super(RamjetBin7, self).__init__()
    
        self.input_dim = input_dim
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

        if self.input_dim == 2700:
            self.block8 = DenseBlock(input_dim=128*17, output_dim=512, dropout=self.dropout)
        elif self.input_dim == 2500:
            self.block8 = DenseBlock(input_dim=128*15, output_dim=512, dropout=self.dropout)
        else:
            raise ValueError('input_dim not supported')
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


class ResNetBin7(nn.Module):
    """1D CNN Architecture with residual connections
    """

    def __init__(self, input_dim=2700, output_dim=1, dropout=0.1):
        super(ResNetBin7, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.block0 = ConvResBlock(in_channels=1, out_channels=8, kernel_size=3, stride=1, batch_normalization=False,
                                                dropout=0)
        self.block0a = ConvResBlock(in_channels=8, out_channels=8, kernel_size=3, stride=2, batch_normalization=False,
                                                dropout=0)
        self.block1 = ConvResBlock(in_channels=8, out_channels=8, kernel_size=3, stride=2, dropout=self.dropout)
        self.block2 = ConvResBlock(in_channels=8, out_channels=16, kernel_size=3, stride=2, dropout=self.dropout)
        self.block3 = ConvResBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2, dropout=self.dropout)
        self.block4 = ConvResBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, dropout=self.dropout)
        self.block5 = ConvResBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, dropout=self.dropout)
        self.block6 = ConvResBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, dropout=self.dropout) # another pool
        self.block7 = ConvResBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, dropout=self.dropout)

        if self.input_dim == 2700:
            self.block8 = DenseBlock(input_dim=128*22, output_dim=512, dropout=self.dropout)
        elif self.input_dim == 2500:
            self.block8 = DenseBlock(input_dim=128*20, output_dim=512, dropout=self.dropout)
        else:
            raise ValueError('input_dim not supported')
        self.block9 = DenseBlock(input_dim=512, output_dim=20, dropout=0, batch_normalization=False)

        self.linear_out = nn.Linear(20, self.output_dim)

    
    def forward(self, x):
        # x: (B, LC_LEN)
        # LC_LEN = 2700 for binned sectors 10-14
        x = x.view(x.shape[0], 1, x.shape[-1])  # input shape: (B, 1, 2700)
        x = self.block0(x)              # (B, 8, 1349)
        x = self.block0a(x)              # (B, 8, 673)
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
