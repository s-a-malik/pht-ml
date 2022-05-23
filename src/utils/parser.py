"""Parse command line arguments.
"""

import sys
import argparse

import torch


def parse_args():
    """Command line argument parser.
    TODO: can use subparsers to group arguments for different modes.
    Returns:
        args: argparse.Namespace object containing the parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="Exoplanet detection from TESS light curves")

    # data config
    parser.add_argument("--data-path",
                        type=str,
                        default="/mnt/zfsusers/shreshth/pht_project/data",
                        help="Root path for data")
    parser.add_argument("--log-dir",
                        type=str,
                        default="/mnt/zfsusers/shreshth/pht_project/code/pht-ml/",
                        help="Root for results/output.")
    parser.add_argument("--synthetic-prob",
                        type=float,
                        default=0.5,
                        help="Augment with synthetic planet data, proportion of data to be synthetic.")
    parser.add_argument("--eb-prob",
                        type=float,
                        default=0.0,
                        help="Augment with eclipsing binary data, proportion of data to be EB injected.")
    parser.add_argument("--num-workers",
                        type=int,
                        default=0,
                        help="number of data loading workers") 
    parser.add_argument("--no-cache",
                        action="store_true",
                        help="Do not use cache for dataloader")
    parser.add_argument("--bin-factor",
                        type=int,
                        default=7,
                        help="binning factor for light curves")
    parser.add_argument("--aug-prob",
                        type=float,
                        default=0.0,
                        help="Probability of augmenting data with random defects.")
    parser.add_argument("--permute-fraction",
                        type=float,
                        default=0.0,
                        help="Fraction of light curve to be randomly permuted.")
    parser.add_argument("--delete-fraction",
                        type=float,
                        default=0.0,
                        help="Fraction of light curve to be randomly deleted.")
    parser.add_argument("--multi-transit",
                        action="store_true",
                        help="take all transits in light curve from simulated data.")
                
    # model config
    parser.add_argument("--model",
                        type=str,
                        default="ramjet",
                        help="Model type: (ramjet, dense,).")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.1,
                        help="Dropout rate.")
    parser.add_argument("--hid-dims",
                        type=int,
                        nargs="*",
                        default=[64],
                        help="Hidden layer dimensions, takes multiple arguments e.g. --hid-dims 64 32")
    parser.add_argument("--activation",
                        type=str,
                        default="ReLU",
                        help="Activation function.")
    parser.add_argument("--kernel-size",
                        type=int,
                        default=3,
                        help="Kernel size for convolutional layers.")
    parser.add_argument("--num-layers",
                        type=int,
                        default=1,
                        help="Number of convolutional layers.")
    # parser.add_argument("--max-lc-length",
    #                     type=int,
    #                     default=2600,
    #                     help="Maximum length of light curve. Default to 18900/7 for binned flux sectors 10-14")


    # training config
    parser.add_argument("--optimizer",
                        type=str,
                        default="adam",
                        help="optimizer")
    parser.add_argument("--loss",
                        type=str,
                        default="BCE",
                        help="loss function")
    parser.add_argument("--disable-cuda",
                        action="store_true",
                        help="don't use GPU")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed")
    parser.add_argument("--batch-size",
                        type=int,
                        default=256,
                        help="batch size")
    parser.add_argument("--epochs",
                        type=int,
                        default=500,
                        help="number of epochs")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-3,
                        help="learning rate")
    parser.add_argument("--weight-decay",
                        type=float,
                        default=0.0,
                        help="weight decay")
    parser.add_argument("--momentum",
                        type=float,
                        default=0.9,
                        help="momentum")
    parser.add_argument("--patience",
                        type=int,
                        default=100000,
                        help="number of epochs patience")
    
    # evaluation
    parser.add_argument("--evaluate",
                        action="store_true",
                        help="evaluate model on test set (don't train")
    parser.add_argument("--checkpoint",
                        type=str,
                        default="",
                        help="wandb run id to load model from")
    parser.add_argument("--val-size",
                        type=float,
                        default=0.2,
                        help="proportion of data to use for validation")
    parser.add_argument("--test-size",
                        type=float,
                        default=0.2,
                        help="proportion of data to use for testing")


    # wandb config
    parser.add_argument("--wandb-offline",
                        action="store_true",
                        help="don't use wandb")
    parser.add_argument("--wandb-project",
                        type=str,
                        default="pht-ml",
                        help="wandb project name")
    parser.add_argument("--experiment-name",
                        type=str,
                        default="",
                        help="wandb experiment name")



    args = parser.parse_args(sys.argv[1:])

    args.device = torch.device("cuda") if (not args.disable_cuda) and \
        torch.cuda.is_available() else torch.device("cpu")

    return args
