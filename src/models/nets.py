"""model_classes.py
Model classes for light curve classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightCurveNetworkBlock(nn.Module):
    """A block containing a convolution and all the fixings that go with it.
    Adapted from ramjet https://github.com/golmschenk/ramjet/blob/master/ramjet/models/components/light_curve_network_block.py
    N.B. removed spatial dropout.
    TODO port from keras
    """
    def __init__(self, filters: int, kernel_size: int, pooling_size: int, dropout: float = 0.1,
                 batch_normalization: bool = True, non_linearity: str = "LeakyReLU"):
        super(LightCurveNetworkBlock, self).__init__()
        self.convolution = nn.Conv1D(filters, kernel_size=kernel_size)
        self.act = get_activation(activation)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        if pooling_size > 1:
            self.max_pooling = nn.MaxPool1D(kernel_size=pooling_size)
        else:
            self.max_pooling = None
        if batch_normalization:
            self.batch_normalization = nn.BatchNorm1d(filters)
            self.batch_normalization_input_reshape = Reshape([-1])
            self.batch_normalization_output_reshape = Reshape([-1, filters])
        else:
            self.batch_normalization = None
#convolution/dense transformation, activation, dropout, max pooling, and batch normalization.

    def call(self, x):
        """
        The forward pass of the layer.
        """
        x = self.convolution(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.max_pooling is not None:
            x = self.max_pooling(x)
        if self.batch_normalization is not None:
            if self.batch_normalization_input_reshape is not None:
                x = self.batch_normalization_input_reshape(x)
            x = self.batch_normalization(x)
            if self.batch_normalization_output_reshape is not None:
                x = self.batch_normalization_output_reshape(x)
        return x




class Ramjet(nn.Module):
    """1D CNN Architecture from 
    Identifying Planetary Transit Candidates in TESS Full-frame Image Light Curves via Convolutional Neural Networks, Olmschenk 2021
    https://iopscience.iop.org/article/10.3847/1538-3881/abf4c6
    """
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        super(Ramjet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

    def __init__(self):
        super().__init__()

        self.block0 = LightCurveNetworkBlock(filters=8, kernel_size=3, pooling_size=2, batch_normalization=False,
                                             dropout=0)
        self.block1 = LightCurveNetworkBlock(filters=8, kernel_size=3, pooling_size=2, dropout=self.dropout)
        self.block2 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=2, dropout=self.dropout)
        self.block3 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=2, dropout=self.dropout)
        self.block4 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, dropout=self.dropout)
        self.block5 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, dropout=self.dropout)
        self.block6 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=1, dropout=self.dropout)
        self.block7 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=1, dropout=self.dropout)
        self.block8 = LightCurveNetworkBlock(filters=20, kernel_size=3, pooling_size=1, dropout=self.dropout)
        self.block9 = LightCurveNetworkBlock(filters=20, kernel_size=7, pooling_size=1, dropout=self.dropout)
        self.block10 = LightCurveNetworkBlock(filters=20, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout=0)

        # TODO add dense layers as in paper

                                            
        self.prediction_layer = nn.Conv1d(1, out_channels, kernel_size=1)
        # TODO port from keras
        self.reshape = Reshape([1])

    def forward(self, x):
        # LC_LEN = 2900 for binned sectors 10-14
        x = x.view(-1, 1, self.input_size)
        # input shape: (B, 1, LC_LEN)
        x = self.block0(x)              # (B, 8, LC_LEN)
        x = self.block1(x)              # (batch_size, 1, 8)
        x = self.block2(x) 
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.prediction_layer(x)
        x = F.sigmoid(x)
        outputs = self.reshape(x)
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
