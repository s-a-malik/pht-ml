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
    parser.add_argument("--lc-root-path",
                        type=str,
                        default="/mnt/zfsusers/shreshth/pht_project/data/TESS",
                        help="Root data directory for light curves")
    parser.add_argument("--labels-root-path",
                        type=str,
                        default="/mnt/zfsusers/shreshth/pht_project/data/pht_labels",
                        help="Root path for labels")
    parser.add_argument("--log-dir",
                        type=str,
                        default="/mnt/zfsusers/shreshth/pht_project/code/pht-ml/",
                        help="Root for results/output.")
    parser.add_argument("--synthetic-prop",
                        type=float,
                        default=0.0,
                        help="Augment with synthetic data, proportion of data to be synthetic.")
    parser.add_argument("--num-workers",
                        type=int,
                        default=0,
                        help="number of data loading workers") 
    parser.add_argument("--bin-factor",
                        type=int,
                        default=1,
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

                
    # model config
    parser.add_argument("--model",
                        type=str,
                        default="dense",
                        help="Model type.")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.0,
                        help="Dropout rate.")
    parser.add_argument("--hid-dims",
                        type=int,
                        nargs="*",
                        default=[64],
                        help="Hidden layer dimensions, takes multiple arguments")
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
                        default=64,
                        help="batch size")
    parser.add_argument("--epochs",
                        type=int,
                        default=50,
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
                        default=10,
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
