"""General utilities
"""

import os
import shutil

import wandb

import torch

from models import nets

SHORTEST_LC = 18900 # sectors 10-14
# SHORTEST_LC = 17500 # sectors 10-38

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


def init_optim(args, model):
    """Initialize optimizer and loss function
    Params:
    - args (argparse.Namespace): parsed command line arguments
    - model (nn.Module): initialised model
    Returns:
    - optimizer (nn.optim): initialised optimizer
    - criterion: initialised loss function
    """
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        raise NameError(f"Unknown optimizer {args.optimizer}")
    
    if args.loss == "BCE":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.loss == "MSE":
        criterion = torch.nn.MSELoss()
    else:
        raise NameError(f"Unknown loss function {args.loss}")

    return optimizer, criterion


def init_model(args):
    """Initialize model
    """
    if args.model == "dense":
        model = nets.SimpleNetwork(
            input_dim=int(SHORTEST_LC / args.bin_factor),
            hid_dims=args.hid_dims,
            output_dim=1,
            non_linearity=args.activation,
            dropout=args.dropout
        )
    elif args.model == "ramjet":
        if args.bin_factor == 3:
            model = nets.RamjetBin3(
                input_dim=int(SHORTEST_LC / args.bin_factor),
                output_dim=1,
                dropout=0.1
            )
        elif args.bin_factor == 7:
            model = nets.RamjetBin7(
                input_dim=int(SHORTEST_LC / args.bin_factor),
                output_dim=1,
                dropout=0.1
            )
    else:
        raise NameError(f"Unknown model {args.model}")
    model.to(args.device)

    return model


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

    if is_best:
        shutil.copyfile(checkpoint_file, best_file)


def plot_lc(x, save_path="/mnt/zfsusers/shreshth/pht_project/data/examples/test_light_curve.png"):
    """Plot light curve for debugging
    Params:
    - x (np.array): light curve
    """

    # plot it
    fig, ax = plt.subplots(figsize=(16, 5))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)

    ## plot the binned and unbinned LC
    ax.plot(list(range(len(x))), x,
        color="royalblue",
        marker="o",
        markersize=1,
        lw=0,
        label="unbinned",
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

    ax.tick_params(axis="y", direction="in", pad=-30, color="white", labelcolor="white")
    ax.tick_params(axis="x", direction="in", pad=-17, color="white", labelcolor="white")

    # ax.set_xlabel("Time (days)", fontsize=10, color="white")

    ax.set_facecolor("#03012d")

    ## save the image
    if save_path is not None:
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none")
    # plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    return fig, ax
