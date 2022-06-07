"""Metrics for classification and regression
"""

import numpy as np
import torch


def bce_loss_numpy(preds, labels, reduction="none", eps=1e-7):
    """Computes the binary cross-entropy loss between predictions and labels.
    Params:
    - preds (np.array): predictions
    - labels (np.array): labels
    - reduction (str): reduction type
    - eps (float): small number to avoid log(0)
    Returns:
    - loss (float): binary cross-entropy loss
    """
    preds = np.clip(preds, eps, 1 - eps)
    if reduction == "none":
        return -labels * np.log(preds) - (1 - labels) * np.log(1 - preds)
    elif reduction == "mean":
        return -np.mean(labels * np.log(preds) + (1 - labels) * np.log(1 - preds))
    elif reduction == "sum":
        return -np.sum(labels * np.log(preds) + (1 - labels) * np.log(1 - preds))
    else:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    
