"""model_classes.py
Model classes for light curve classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# class ConvBlock(nn.Module):
#     """Single convolutional block with all the trimmings
#     """
#     def __init__(self):
#         super(ConvBlock, self).__init__()



# class Ramjet(nn.Module):
#     """Architecture from 
#     Identifying Planetary Transit Candidates in TESS Full-frame Image Light Curves via Convolutional Neural Networks, Olmschenk 2021
#     https://iopscience.iop.org/article/10.3847/1538-3881/abf4c6
#     """
#     def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
#         super(Ramjet, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.dropout = dropout

#     def forward(self, x):
#         pass


class SimpleNetwork(nn.Module):
    """Generic fully connected MLP with adjustable depth.
    """
    def __init__(self, input_dim, hid_dims, output_dim, non_linearity="ReLU", dropout=0.0):
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
       
       if non_linearity == "ReLU":
            self.act = nn.ReLU()
        elif non_linearity == "tanh":
            self.act = nn.Tanh()
        elif non_linearity == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NameError(f"activation {non_linearity} not defined")
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
    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)
        """
        super(ResidualNetwork, self).__init__()

        dims = [input_dim]+hidden_layer_dims

        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])

        self.res_fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False)
                                      if (dims[i] != dims[i+1])
                                      else nn.Identity()
                                      for i in range(len(dims)-1)])

        self.acts = nn.ModuleList([nn.ReLU() for _ in range(len(dims)-1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea))+res_fc(fea)

        return self.fc_out(fea)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

