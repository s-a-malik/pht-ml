"""main.py
Primary entry point for the pht-ml package.
run on cluster with GPU e.g.:  addqueue -c "comment" -m 4 -n 1x4 -q gpulong -s ../shell_scripts/run_batch_experiments.sh
"""
import os
import wandb

import numpy as np
import pandas as pd

import torch
torch.multiprocessing.set_sharing_strategy('file_system')   # fix memory leak?

from utils.utils import load_checkpoint
from utils.parser import parse_args
from utils.data import get_data_loaders
from models.train import training_run, evaluate, init_model, init_optim


def main(args):
    
    # random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # init directories
    results_path = f"{args.log_dir}/results"
    os.makedirs(results_path, exist_ok=True)

    # init wandb
    os.environ['WANDB_MODE'] = 'offline' if args.wandb_offline else 'online' 
    # os.environ['WANDB_MODE'] = 'offline'
    # change artifact cache directory to scratch
    os.environ['WANDB_CACHE_DIR'] = os.getenv('SCRATCH_DIR', './')
    job_type = "eval" if args.evaluate else "train"
    run = wandb.init(entity="s-a-malik",
                     project=args.wandb_project,
                     group=args.experiment_name,
                     job_type=job_type,
                     settings=wandb.Settings(start_method="fork"),   # this is to prevent InitStartError
                     save_code=True)
    wandb.config.update(args)

    # initialise models, optimizers, data
    model = init_model(args)
    optimizer, criterion = init_optim(args, model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # number of model parameters
    print(f"Number of model parameters: {num_params}")
    wandb.config.num_params = num_params     # add to wandb config
    print(model)
    print(optimizer)
    print(criterion)

    train_loader, val_loader, test_loader = get_data_loaders(args)
    wandb.config.num_train_examples = len(train_loader.dataset)
    wandb.config.train_sectors = train_loader.dataset.sectors
    wandb.config.num_val_examples = len(val_loader.dataset)
    wandb.config.val_sectors = val_loader.dataset.sectors
    wandb.config.num_test_examples = len(test_loader.dataset)
    wandb.config.test_sectors = test_loader.dataset.sectors

    # files for checkpoints
    scratch_dir = os.getenv('SCRATCH_DIR', wandb.run.dir)   # if given a scratch dir save models here
    best_file = os.path.join(scratch_dir, "best.pth.tar")  

    # load previous state
    if args.checkpoint:
        model_path = f"{args.log_dir}/checkpoints/{args.checkpoint}"
        os.makedirs(model_path, exist_ok=True)
        # restore from wandb
        wandb_best_file = wandb.restore(
            "best.pth.tar",
            run_path=f"s-a-malik/{args.wandb_project}/{args.checkpoint}",
            root=model_path)
        # load state dict
        model, optimizer = load_checkpoint(model, optimizer, args.device,
                                                 wandb_best_file.name)
    
    # train
    if not args.evaluate:
        model = training_run(args, model, optimizer, criterion, train_loader, val_loader)
    else:
        best_file = wandb_best_file.name
    
    # load model
    model, optimizer = load_checkpoint(model, optimizer, args.device, best_file)

    # evaluate on test set
    with torch.no_grad():
        test_loss, test_acc, test_f1, test_prec, test_rec, test_auc, results = evaluate(model, optimizer, criterion, test_loader, args.device, task="test")
    
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test precision: {test_prec:.4f}, Test recall: {test_rec:.4f}, Test AUC: {test_auc:.4f}")
    
    results = {"test/" + k: v for k, v in results.items()}
    results.update({"test/loss": test_loss,
        "test/acc": test_acc,
        "test/f1": test_f1,
        "test/prec": test_prec,
        "test/rec": test_rec,
        "test/auc": test_auc})
    wandb.log(results)

    # save to a results file
    # df = pd.DataFrame({"pred": test_pred, "targets": test_targets, "tics": test_tics, "secs": test_secs, "tic_injs": test_tic_injs})
    # df.to_csv(f"{results_path}/results.csv", index=False)

    # finish wandb
    run.finish()


if __name__ == "__main__":
    args = parse_args()
    # TODO experiment yaml config file instead?
    print("running on {}".format(args.device))
    print(args)
    main(args)
