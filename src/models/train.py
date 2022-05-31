"""train.py
Utility functions and classes for model training and evaluation.
"""

import os
import time
import datetime

import matplotlib.pyplot as plt

from tqdm.autonotebook import trange

import wandb

import torch

import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

from utils import utils
from utils.data import SHORTEST_LC
from models import nets

def evaluate(model, optimizer, criterion, data_loader, device, task="train", save_examples=-1):
    """Run one batch through model
    Params:
    - optimizer (nn.optim): optimizer tied to model weights. 
    - criterion: loss function
    - data_loader (torch.utils.data.DataLoader): data loader
    - device (torch.device): cuda or cpu
    - task (str): train, val, test
    - save_examples (int): whether to save example predictions (-1 for no, epoch number for yes)
    Returns:
    - loss: loss on batch
    - acc: accuracy on batch
    - in addition, if test:
        - preds: list of class predictions
        - test_targets: list of true labels
        - test_tics: list of tic_ids of the test batch
        - test_secs: list of sectors of the test batch
    """
    avg_loss = utils.AverageMeter()
    targets = []
    targets_bin = []
    probs = []
    preds = []
    tics = []
    secs = []
    tic_injs = []
    snrs = []
    if save_examples != -1:
        fluxs = []
    total = 0
    if task in ["val", "test"]:
        model.eval()
    elif task == "train":
        model.train()
    else:
        raise NameError("Only train, val or test is allowed as task")
    
    # pytorch profiler
    # with torch.profiler.profile(
    # schedule=torch.profiler.schedule(
    #     wait=2,
    #     warmup=2,
    #     active=6,
    #     repeat=1),
    # on_trace_ready=torch.profiler.tensorboard_trace_handler,
    # with_stack=True
    # ) as profiler:

    with trange(len(data_loader)) as t:
        for i, batch in enumerate(data_loader):
            # unpack batch from dataloader
            x, y = batch
            flux = x["flux"]
            flux = flux.to(device)
            y = y.to(device)
            logits = model(flux)
            prob = torch.sigmoid(logits)
            # preds = np.where(probs > 0.5, 1, 0)
            pred = (prob > 0.5).float()
            y_bin = (y > 0.5).float()

            # compute loss on logits
            loss = criterion(logits, torch.unsqueeze(y, 1))
            avg_loss.update(loss.data.cpu().item(), y.size(0))     
            
            if task == "train":
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if save_examples != -1:
                flux = flux.detach().cpu().numpy()
                fluxs += flux.tolist()

            prob = prob.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()
            y_bin = y_bin.detach().cpu().numpy()
            # collect the model outputs
            targets += y.tolist()
            targets_bin += y_bin.tolist()
            probs += np.squeeze(prob).tolist()
            preds += np.squeeze(pred).tolist()
            tics += x["tic"].tolist()
            secs += x["sec"].tolist()
            tic_injs += x["tic_inj"].tolist()
            snrs += x["snr"].tolist()
            total += logits.size(0)

            t.update()

            # profiler.step()
            
    # print("targets", targets)
    # print("pred probs", probs)
    acc = accuracy_score(targets_bin, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(targets_bin, preds, average="binary")
    auc = roc_auc_score(targets_bin, probs)

    # save example predictions to wandb for inspection
    if save_examples != -1:
        utils.save_examples(fluxs, preds, targets, targets_bin, tics, secs, tic_injs, snrs, save_examples)

    if task == "test":
        return avg_loss.avg, acc, f1, prec, rec, auc, probs, targets, tics, secs, tic_injs, total
    else:
        return avg_loss.avg, acc, f1, prec, rec, auc


def training_run(args, model, optimizer, criterion, train_loader, val_loader):
    """Run training loop
    Params:
    - args (argparse.Namespace): parsed command line arguments
    - model (nn.Module): model to train
    - optimizer (nn.optim): optimizer tied to model weights.
    - criterion: loss function
    - train_loader (torch.utils.data.DataLoader): training data loader
    - val_loader (torch.utils.data.DataLoader): validation data loader
    Returns:
    - model (nn.Module): trained model
    """

    # get best val loss
    best_loss, best_acc, _, _, _, _ = evaluate(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                data_loader=val_loader,
                device=args.device,
                task="val",
                save_examples=-1 if args.example_save_freq == -1 else 0)
    print(f"\ninitial loss: {best_loss}, acc: {best_acc}")
    best_epoch = 0
    start_time = time.time()
    try:
        # Training loop
        for epoch in range(1, args.epochs):
            epoch_start = time.time()
            train_loss, train_acc, train_f1, train_prec, train_rec, train_auc = evaluate(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                data_loader=train_loader,
                device=args.device,
                task="train")

            # evaluate on val set
            if (args.example_save_freq != -1) and ((epoch) % args.example_save_freq == 0):
                save_examples = epoch
                print("saving example predictions on val set")
            else:
                save_examples = -1
            val_loss, val_acc, val_f1, val_prec, val_rec, val_auc = evaluate(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                data_loader=val_loader,
                device=args.device,
                task="val",
                save_examples=save_examples)

            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                best_epoch = epoch
            wandb.log(
                {
                    "train/acc": train_acc,
                    "train/f1": train_f1,
                    "train/prec": train_prec,
                    "train/rec": train_rec,
                    "train/auc": train_auc,
                    "train/loss": train_loss,
                    "val/acc": val_acc,
                    "val/f1": val_f1,
                    "val/prec": val_prec,
                    "val/rec": val_rec,
                    "val/auc": val_auc,
                    "val/loss": val_loss,
                })

            # save checkpoint
            checkpoint_dict = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
                "args": vars(args)
            }
            utils.save_checkpoint(checkpoint_dict, is_best)

            print(f"\nTime for Epoch: {time.time() - epoch_start:.2f}s, Total Time: {time.time() - start_time:.2f}s"
                f"\nEpoch {epoch+1}/{args.epochs}: \ntrain/loss: {train_loss}, train/acc: {train_acc}, train/prec: {train_prec} train/rec: {train_rec}, train/f1: {train_f1}, train/auc: {train_auc}"
                f"\nval/loss: {val_loss}, val/acc: {val_acc}, val/prec: {val_prec}, val/rec: {val_rec}, val/f1: {val_f1}, val/auc: {val_auc}"
            )

            # patience
            if (epoch - best_epoch > args.patience):
                print("\nEarly stopping...")
                break
    
    except KeyboardInterrupt:
        pass

    return model


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
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        raise NameError(f"Unknown optimizer {args.optimizer}")
    
    if args.loss == "BCE":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.loss == "MSE":
        criterion = torch.nn.MSELoss()
    else:
        raise NameError(f"Unknown loss function {args.loss}")

    return optimizer, criterion
