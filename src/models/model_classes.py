"""model_classes.py
Model classes for light curve classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCNN(nn.Module):
    """TODO
    """
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        super(BaseCNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, x):
        pass
