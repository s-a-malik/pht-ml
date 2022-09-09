"""General utilities
"""

import os
import shutil
from ast import literal_eval

import wandb

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator

import pandas as pd
import numpy as np

import torch

TRAIN_SECTORS_DEBUG = [10]
TRAIN_SECTORS_STANDARD = [10,11,12,13,14,15,16,17,18,19,20]
TRAIN_SECTORS_FULL = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
TRAIN_SECTORS_ALL = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

# keep validation and test sets the same
VAL_SECTORS_DEBUG = [12]
VAL_SECTORS_STANDARD = [30,31,32,33,34,35]
VAL_SECTORS_FULL = [30,31,32,33,34,35]
VAL_SECTORS_ALL = [30,31,32,33,34,35]

TEST_SECTORS_DEBUG = [14]
TEST_SECTORS_STANDARD = [36,37,38]
TEST_SECTORS_FULL = [36,37,38]
TEST_SECTORS_ALL = [36,37,38,39,40,41,42,43]

SHORTEST_LC = 17500 # from sector 10-38. Used to trim all the data to the same length.

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_checkpoint(model, optimizer, device, checkpoint_file: str):
    """Loads a model checkpoint.
    Params:
    - model (nn.Module): initialised model
    - optimizer (nn.optim): initialised optimizer
    - device (torch.device): device model is on
    Returns:
    - model with loaded state dict
    - optimizer with loaded state dict
    """
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Loaded {checkpoint_file}, "
          f"trained to epoch {checkpoint['epoch']+1} with best loss {checkpoint['best_loss']}")

    return model, optimizer, checkpoint["epoch"]+1, checkpoint["best_loss"]


def save_checkpoint(checkpoint_dict: dict, is_best: bool):
    """Saves a model checkpoint to file. Keeps most recent and best model.
    Params:
    - checkpoint_dict (dict): dict containing all model state info, to pickle
    - is_best (bool): whether this checkpoint is the best seen so far.
    """
    # files for checkpoints
    scratch_dir = os.getenv('SCRATCH_DIR', wandb.run.dir)   # if given a scratch dir save models here
    checkpoint_file = os.path.join(scratch_dir, "ckpt.pth.tar")
    best_file = os.path.join(scratch_dir, "best.pth.tar")  
    torch.save(checkpoint_dict, checkpoint_file)
    wandb.save(checkpoint_file, policy="live")  # save to wandb
    print(f"Saved checkpoint to {checkpoint_file}")

    if is_best:
        shutil.copyfile(checkpoint_file, best_file)
        print(f"Saved best checkpoint to {best_file}")
        wandb.save(best_file, policy="live")    # save to wandb


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


def plot_lc(x, save_path=None):
    """Plot light curve for debugging
    Params:
    - x (np.array): light curve
    """

    # close all previous figures
    plt.close('all')

    # plot it
    fig, ax = plt.subplots(figsize=(16, 5))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)

    ## plot the binned and unbinned LC
    ax.plot(list(range(len(x))), x,
        color="royalblue",
        marker="o",
        markersize=1,
        lw=0,
    )
    ## label the axis.
    ax.xaxis.set_label_coords(0.063, 0.06)  # position of the x-axis label

    ## define tick marks/axis parameters
    minorLocator = AutoMinorLocator()
    ax.xaxis.set_minor_locator(minorLocator)
    ax.tick_params(direction="in", which="minor", colors="w", length=3, labelsize=13)

    minorLocator = AutoMinorLocator()
    ax.yaxis.set_minor_locator(minorLocator)
    ax.tick_params(direction="in", length=3, which="minor", colors="grey", labelsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    ax.tick_params(axis="y", direction="in", pad=-50, color="white", labelcolor="white")
    ax.tick_params(axis="x", direction="in", pad=-17, color="white", labelcolor="white")

    # ax.set_xlabel("Time (days)", fontsize=10, color="white")

    ax.set_facecolor("#03012d")

    ## save the image
    if save_path is not None:
        plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
        print(f"Saved light curve to {save_path}")

    return fig, ax


def save_examples(results, step):
    """Plot example predictions for inspection and save to wandb
    Params:
    - results (dict):
        - fluxs (list): predicted fluxes
        - probs (list): predicted probabilities
        - targets (list): true fluxes
        - targets_bin (list): true binary targets
        - tics (list): tic ids
        - secs (list): secs
        - tic_injs (list): tic injections
        - snrs (list): snrs
        - durations (list): durations
        - periods (list): periods
        - depths (list): depths
        - eb_prim_depths (list): primary eb depth
        - eb_sec_depths (list): secondary eb depth
        - eb_periods (list): eb periods
    - step (int): step number (epoch)
    """
    probs = np.array(results["probs"])
    targets = np.array(results["targets"])

    # most confident preds
    conf_preds_sorted = np.argsort(probs)[::-1]
    conf_preds_sorted = conf_preds_sorted[:5]
    for i, idx in enumerate(conf_preds_sorted):
        plt.clf()
        fig, ax = plot_lc(results["fluxs"][idx])
        _set_title(results, idx, ax)
        wandb.log({f"pos_preds_{i}": wandb.Image(fig)}, step=step)

    # confident negative preds
    neg_preds_sorted = np.argsort(probs)
    neg_preds_sorted = neg_preds_sorted[:5]
    for i, idx in enumerate(neg_preds_sorted):
        plt.clf()
        fig, ax = plot_lc(results["fluxs"][idx])
        _set_title(results, idx, ax)        
        wandb.log({f"neg_preds_{i}": wandb.Image(fig)}, step=step)

    # most uncertain preds (closest to 0.5)
    unc_preds_sorted = np.argsort(np.abs(0.5 - probs))
    unc_preds_sorted = unc_preds_sorted[:5]
    for i, idx in enumerate(unc_preds_sorted):
        plt.clf()
        fig, ax = plot_lc(results["fluxs"][idx])
        _set_title(results, idx, ax)
        wandb.log({f"unc_preds_{i}": wandb.Image(fig)}, step=step)

    # most lossy preds (highest difference between prob and target)
    loss_preds_sorted = np.argsort(np.abs(probs - targets))[::-1]
    loss_preds_sorted = loss_preds_sorted[:5]
    for i, idx in enumerate(loss_preds_sorted):
        plt.clf()
        fig, ax = plot_lc(results["fluxs"][idx])
        _set_title(results, idx, ax)   
        wandb.log({f"worst_preds_{i}": wandb.Image(fig)}, step=step)

    # losses
    bce_losses = bce_loss_numpy(probs, targets)
    # log results to wandb to be plotted in the dashboard (without flux)
    df = pd.DataFrame({"bce_loss": bce_losses, "prob": probs, "target": targets, 
                    "class": results["classes"], "tic": results["tics"], "sec": results["secs"], 
                    "toi": results["tois"], "tce": results["tces"], "ctc": results["ctcs"], "ctoi": results["ctois"],
                    "tic_inj": results["tic_injs"], "snr": results["snrs"], "duration": results["durations"], "period": results["periods"], "depth": results["depths"],
                    "eb_prim_depth": results["eb_prim_depths"], "eb_sec_depth": results["eb_sec_depths"], "eb_period": results["eb_periods"],
                    "tic_noise": results["tic_noises"]})

    wandb.log({"val/results": wandb.Table(dataframe=df),
                "val/roc": wandb.plot.roc_curve(np.array(results["targets_bin"], dtype=int), np.stack((1-probs,probs),axis=1)),
                "val/pr": wandb.plot.pr_curve(np.array(results["targets_bin"], dtype=int), np.stack((1-probs,probs),axis=1))},
                step=step)


def _set_title(results, idx, ax):
    """Set the title of the plot 
    """
    if results["snrs"][idx] != -1:
        # this is an injected planet 
        ax.set_title(f'TRANSIT: tic: {results["tics"][idx]} sec: {results["secs"][idx]} tic_inj: {results["tic_injs"][idx]}, snr: {results["snrs"][idx]} prob: {results["probs"][idx]}, target: {results["targets"][idx]}')
    elif results["eb_periods"][idx] != -1:
        # this is an injected eclipsing binary
        ax.set_title(f'EB: tic: {results["tics"][idx]} sec: {results["secs"][idx]} tic_inj: {results["tic_injs"][idx]}, prim_depth: {results["eb_prim_depths"][idx]}, sec_depth: {results["eb_sec_depths"][idx]}, period: {results["eb_periods"][idx]}, prob: {results["probs"][idx]}, target: {results["targets"][idx]}')
    else:
        # this is neither
        ax.set_title(f'tic: {results["tics"][idx]} sec: {results["secs"][idx]} prob: {results["probs"][idx]}, target: {results["targets"][idx]}')


def read_lc_csv(lc_file):
    """Read LC flux from preprocessed csv
    Params:
    - lc_file (str): path to lc_file
    Returns:
    - x (dict): dictionary with keys:
        - flux (np.array): light curve
        - tic (int): TIC
        - sec (int): sector
        - cam (int): camera
        - chi (int): chi
        - tessmag (float): TESS magnitude
        - teff (float): effective temperature
        - srad (float): stellar radius
        - binfac (float): binning factor
        - cdpp(05,1,2) (float): CDPP at 0.5, 1, 2 hour time scales
    """
    try:
        # read the csv file
        df = pd.read_csv(lc_file)
        # get the flux
        x = {}
        x["flux"] = df["flux"].values

        # parse the file name
        file_name = lc_file.split("/")[-1]
        params = file_name.split("_")
        for i, param in enumerate(params):
            if i == len(params) - 1:
                # remove .csv
                x[param.split("-")[0]] = literal_eval(param.split("-")[1][:-4])
            else:
                x[param.split("-")[0]] = literal_eval(param.split("-")[1])
            # convert None to -1
            x[param.split("-")[0]] = -1 if x[param.split("-")[0]] is None else x[param.split("-")[0]]
    except:
        # print("failed to read file: ", lc_file)
        x = {"flux": None}
    return x


def get_sectors(data_split):
    """
    Params:
    - data_split (str): data split to use (train_debug, val_standard etc.)
    Returns:
    - sectors (list): list of sectors
    """
    if data_split == "train_standard":
        return TRAIN_SECTORS_STANDARD
    elif data_split == "val_standard":
        return VAL_SECTORS_STANDARD
    elif data_split == "test_standard":
        return TEST_SECTORS_STANDARD
    if data_split == "train_full":
        return TRAIN_SECTORS_FULL
    elif data_split == "val_full":
        return VAL_SECTORS_FULL
    elif data_split == "test_full":
        return TEST_SECTORS_FULL
    elif data_split == "train_debug":
        return TRAIN_SECTORS_DEBUG
    elif data_split == "val_debug":
        return VAL_SECTORS_DEBUG
    elif data_split == "test_debug":
        return TEST_SECTORS_DEBUG
    elif data_split == "train_all":
        return TRAIN_SECTORS_ALL
    elif data_split == "val_all":
        return VAL_SECTORS_ALL
    elif data_split == "test_all":
        return TEST_SECTORS_ALL
    else:
        raise ValueError(f"Invalid data split {data_split}")
