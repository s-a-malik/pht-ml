"""General utilities
"""

import os
import shutil

import wandb

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator

import pandas as pd
import numpy as np

import torch

from utils import metrics


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

    return model, optimizer


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


def plot_lc(x, save_path="/mnt/zfsusers/shreshth/pht_project/data/examples/test_light_curve.png"):
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
    bce_losses = metrics.bce_loss_numpy(probs, targets)
    # log results to wandb to be plotted in the dashboard (without flux)
    df = pd.DataFrame({"bce_loss": bce_losses, "prob": probs, "target": targets, "tic": results["tics"], "sec": results["secs"], "tic_inj": results["tic_injs"], "snr": results["snrs"], "duration": results["durations"], "period": results["periods"], "depth": results["depths"], "eb_prim_depth": results["eb_prim_depths"], "eb_sec_depth": results["eb_sec_depths"], "eb_period": results["eb_periods"]})

    wandb.log({"val_results": wandb.Table(dataframe=df),
                "roc": wandb.plot.roc_curve(np.array(results["targets_bin"], dtype=int), np.stack((1-probs,probs),axis=1)),
                "pr": wandb.plot.pr_curve(np.array(results["targets_bin"], dtype=int), np.stack((1-probs,probs),axis=1))},
                step=step)


# def plot_scatter_with_hist(x, y, x_label, y_label):
#     """Plot scatter with marginal distributions
#     """
#     plt.clf()
#     # definitions for the axes
#     left, width = 0.1, 0.65
#     bottom, height = 0.1, 0.65
#     spacing = 0.005

#     rect_scatter = [left, bottom, width, height]
#     rect_histx = [left, bottom + height + spacing, width, 0.2]
#     rect_histy = [left + width + spacing, bottom, 0.2, height]

#     # start with a square Figure
#     fig = plt.figure(figsize=(8, 8))

#     ax = fig.add_axes(rect_scatter)
#     ax_histx = fig.add_axes(rect_histx, sharex=ax)
#     ax_histy = fig.add_axes(rect_histy, sharey=ax)

#     # no labels
#     ax_histx.tick_params(axis="x", labelbottom=False)
#     ax_histy.tick_params(axis="y", labelleft=False)

#     # the scatter plot:
#     ax.scatter(x, y)
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)

#     # now determine nice limits by hand:
#     binwidth = 0.25
#     xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
#     lim = (int(xymax/binwidth) + 1) * binwidth

#     bins = np.arange(-lim, lim + binwidth, binwidth)
#     ax_histx.hist(x, bins=bins)
#     ax_histy.hist(y, bins=bins, orientation='horizontal')

#     return fig, ax, ax_histx, ax_histy


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
