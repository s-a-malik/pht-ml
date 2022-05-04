"""main.py
Primary entry point for the pht-ml package.
"""
import os
import sys
import argparse
import wandb

import numpy as np
import pandas as pd

import torch

from utils import utils
from utils.parser import parse_args


def main(args):

    # seeds
    # random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    # initialise directories
    results_path = f"{args.log_dir}/results"
    os.makedirs(results_path, exist_ok=True)

    # wandb
    os.environ['WANDB_MODE'] = 'online' if config['experiment']['log'] else 'offline' 
    # change artifact cache directory to scratch
    os.environ['WANDB_CACHE_DIR'] = os.getenv('SCRATCH_DIR', './') + '.cache/wandb'
    
    job_type = "eval" if args.evaluate else "train"
    run = wandb.init(entity="s-a-malik",
                     project=args.wandb_project,
                     group=args.experiment_name,
                     job_type=job_type,
                     settings=wandb.Settings(start_method="fork")   # this is to prevent InitStartError
                     save_code=True)
    wandb.config.update(args)

    # load previous state
    if args.checkpoint:
        model_path = f"{args.log_dir}/checkpoints/{args.checkpoint}"
        os.makedirs(model_path, exist_ok=True)
        # restore from wandb
        checkpoint_file = wandb.restore(
            "best.pth.tar",
            run_path=f"s-a-malik/{args.wandb_project}/{args.checkpoint}",
            root=model_path)
        # load state dict
        model, optimizer = utils.load_checkpoint(model, optimizer, args.device,
                                                 checkpoint_file.name)

    # train
    model = utils.init_model(args)
    optimizer = utils.init_optim(args)

    # load data


    if not args.evaluate:

        # files for checkpoints
        scratch_dir = os.getenv('SCRATCH_DIR', wandb.run.dir)   # if given a scratch dir save models here
        checkpoint_file = os.path.join(scratch_dir, "acq_model_ckpt.pth.tar")
        best_file = os.path.join(scratch_dir, "acq_model_best.pth.tar")  



    else:
        # load model
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # evaluate
        model.eval()
        model.to(args.device)


    # eval

    # finish
    run.finish()


if __name__ == "__main__":
    args = parse_args()
    # TODO experiment yaml config file instead?

    main(args)
