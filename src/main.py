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
from utils.train import training_run, test_run
from utils.parser import parse_args
from utils.data import get_data_loaders

def main(args):

    # random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    # init directories
    results_path = f"{args.log_dir}/results"
    os.makedirs(results_path, exist_ok=True)

    # init wandb
    os.environ['WANDB_MODE'] = 'offline' if args.wandb_offline else 'online' 
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

    # initialise models, optimizers, data
    model = utils.init_model(args)
    optimizer, criterion = utils.init_optim(args, model)
    train_loader, val_loader, test_loader = get_data_loaders(
        data_root_path=args.lc_root_path,
        labels_root_path=args.labels_root_path,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False
    )

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

    if not args.evaluate:
        # files for checkpoints
        scratch_dir = os.getenv('SCRATCH_DIR', wandb.run.dir)   # if given a scratch dir save models here
        checkpoint_file = os.path.join(scratch_dir, "acq_model_ckpt.pth.tar")
        best_file = os.path.join(scratch_dir, "acq_model_best.pth.tar")  
    
        # train
        # model = training_run(

    
    else:
        # load model
        model, optimizer = load_checkpoint(model, optimizer, args.device, checkpoint_file):


    # evaluate on test set
    test_loss, test_acc = test_run(model, test_loader)


    # finish
    run.finish()


if __name__ == "__main__":
    args = parse_args()
    # TODO experiment yaml config file instead?

    main(args)
