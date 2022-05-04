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
    parser.add_argument("--data-dir",
                        type=str,
                        default="/mnt/zfsusers/shreshth/pht_project/data/",
                        help="Root data directory.")
    parser.add_argument("--file-list",
                        type=str,
                        default="",
                        help="File containing list of light curve file paths.")
    parser.add_argument("--log-dir",
                        type=str,
                        default="/mnt/zfsusers/shreshth/pht_project/code/pht-ml/",
                        help="Root for results/output.")
    parser.add_argument("--synthetic-prop",
                        type=float,
                        default=0.0,
                        help="Augment with synthetic data, proportion of data to be synthetic.")
    

                
    # model config



    # training config
    parser.add_argument("--disable-cuda",
                        action="store_true",
                        help="don't use GPU")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed")
    
    # evaluation
    parser.add_argument("--evaluate",
                        action="store_true",
                        help="evaluate model on test set (don't train")
    parser.add_argument("--checkpoint",
                        type=str,
                        default="",
                        help="wandb run id to load model from")



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
