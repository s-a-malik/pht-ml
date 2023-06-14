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

from utils.utils import load_checkpoint, bce_loss_numpy
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
    # change artifact cache directory to scratch
    os.environ['WANDB_CACHE_DIR'] = os.getenv('SCRATCH_DIR', './')
    job_type = "eval" if args.evaluate else "train"
    run = wandb.init(entity=args.wandb_entity,
                     project=args.wandb_project,
                     group=args.experiment_name,
                     job_type=job_type,
                     settings=wandb.Settings(start_method="fork"),   # this is to prevent InitStartError
                     save_code=True)
    wandb.config.update(args)

    # initialise models, optimizers, data
    model = init_model(args)
    optimizer, criterion, scheduler = init_optim(args, model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # number of model parameters
    print(f"Number of model parameters: {num_params}")
    print(model)
    print(optimizer)
    print(scheduler)
    print(criterion)

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
        model, optimizer, scheduler, best_epoch, _ = load_checkpoint(model, optimizer, scheduler, args.device,
                                                 wandb_best_file.name)

    # get data
    train_loader, val_loader, test_loader = get_data_loaders(args)

    # add to wandb config
    wandb.config.num_params = num_params     
    wandb.config.num_train_examples = len(train_loader.dataset)
    wandb.config.train_sectors = train_loader.dataset.sectors
    wandb.config.num_val_examples = len(val_loader.dataset)
    wandb.config.val_sectors = val_loader.dataset.sectors
    wandb.config.num_test_examples = len(test_loader.dataset)
    wandb.config.test_sectors = test_loader.dataset.sectors

    # train
    if not args.evaluate:
        model, epoch = training_run(args, model, optimizer, scheduler, criterion, train_loader, val_loader)
    else:
        epoch = None
        best_file = wandb_best_file.name
    
    # load model
    model, optimizer, scheduler, best_epoch, _ = load_checkpoint(model, optimizer, scheduler, args.device, best_file)

    # evaluate on all sets
    with torch.no_grad():
        print("\nEVALUATION\n")
        train_loss, train_acc, train_f1, train_prec, train_rec = evaluate(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                data_loader=train_loader,
                device=args.device,
                task="val")
        print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, f1: {train_f1:.4f}, precision: {train_prec:.4f}, recall: {train_rec:.4f}")
        val_loss, val_acc, val_f1, val_prec, val_rec, val_auc, val_results = evaluate(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                data_loader=val_loader,
                device=args.device,
                task="test")
        print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}, f1: {val_f1:.4f}, precision: {val_prec:.4f}, recall: {val_rec:.4f}, AUC: {val_auc:.4f}")
        test_loss, test_acc, test_f1, test_prec, test_rec, test_auc, test_results = evaluate(
            model=model, 
            optimizer=optimizer,
            criterion=criterion,
            data_loader=test_loader,
            device=args.device,
            task="test")
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test precision: {test_prec:.4f}, Test recall: {test_rec:.4f}, Test AUC: {test_auc:.4f}")
    # put results in a wandb table
    val_probs = np.array(val_results["probs"])
    val_targets = np.array(val_results["targets"])
    val_bce_losses = bce_loss_numpy(val_probs, val_targets)
    val_df = pd.DataFrame({"bce_loss": val_bce_losses, "prob": val_probs, "target": val_targets, 
                    "class": val_results["classes"], "tic": val_results["tics"], "sec": val_results["secs"], 
                    "toi": val_results["tois"], "tce": val_results["tces"], "ctc": val_results["ctcs"], "ctoi": val_results["ctois"],
                    "tic_inj": val_results["tic_injs"], "snr": val_results["snrs"], "duration": val_results["durations"], "period": val_results["periods"], "depth": val_results["depths"],
                    "eb_prim_depth": val_results["eb_prim_depths"], "eb_sec_depth": val_results["eb_sec_depths"], "eb_period": val_results["eb_periods"],
                    "tic_noise": val_results["tic_noises"]})
    test_probs = np.array(test_results["probs"])
    test_targets = np.array(test_results["targets"])
    test_bce_losses = bce_loss_numpy(test_probs, test_targets)
    test_df = pd.DataFrame({"bce_loss": test_bce_losses, "prob": test_probs, "target": test_targets, 
                    "class": test_results["classes"], "tic": test_results["tics"], "sec": test_results["secs"], 
                    "toi": test_results["tois"], "tce": test_results["tces"], "ctc": test_results["ctcs"], "ctoi": test_results["ctois"],
                    "tic_inj": test_results["tic_injs"], "snr": test_results["snrs"], "duration": test_results["durations"], "period": test_results["periods"], "depth": test_results["depths"],
                    "eb_prim_depth": test_results["eb_prim_depths"], "eb_sec_depth": test_results["eb_sec_depths"], "eb_period": test_results["eb_periods"],
                    "tic_noise": test_results["tic_noises"]})
    wandb.log({
        "train_best/loss": train_loss,
        "train_best/acc": train_acc,
        "train_best/f1": train_f1,
        "train_best/prec": train_prec,
        "train_best/rec": train_rec,
        "val_best/loss": val_loss,
        "val_best/acc": val_acc,
        "val_best/f1": val_f1,
        "val_best/prec": val_prec,
        "val_best/rec": val_rec,
        "val_best/auc": val_auc,
        "val_best/results": wandb.Table(dataframe=val_df),
        "val_best/roc": wandb.plot.roc_curve(np.array(val_results["targets_bin"], dtype=int), np.stack((1-val_probs,val_probs),axis=1)),
        "val_best/pr": wandb.plot.pr_curve(np.array(val_results["targets_bin"], dtype=int), np.stack((1-val_probs,val_probs),axis=1)),
        "test/loss": test_loss,
        "test/acc": test_acc,
        "test/f1": test_f1,
        "test/prec": test_prec,
        "test/rec": test_rec,
        "test/auc": test_auc,
        "test/results": wandb.Table(dataframe=test_df),
        "test/roc": wandb.plot.roc_curve(np.array(test_results["targets_bin"], dtype=int), np.stack((1-test_probs,test_probs),axis=1)),
        "test/pr": wandb.plot.pr_curve(np.array(test_results["targets_bin"], dtype=int), np.stack((1-test_probs,test_probs),axis=1))})

    # finish wandb
    run.finish()


if __name__ == "__main__":
    args = parse_args()
    # TODO experiment yaml config file instead?
    print("running on {}".format(args.device))
    print(args)
    if args.module_test == "dataloader":
        print("\nTESTING DATALOADER")
        from utils.data import test_dataloader
        test_dataloader(args)
    elif args.module_test == "plot":
        print("\nPLOTTING LC")
        from utils.plot_lc import plot_lc_test
        plot_lc_test(args)
    else:
        main(args)
