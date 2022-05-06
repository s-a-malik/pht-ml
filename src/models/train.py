"""train.py
Utility functions and classes for model training and evaluation.
"""

import os

from tqdm.autonotebook import trange

import wandb

import torch

import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

from utils import utils


def evaluate(model, optimizer, criterion, data_loader, device, task="train"):
    """Run one batch through model
    Params:
    - optimizer (nn.optim): optimizer tied to model weights. 
    - criterion: loss function
    - data_loader (torch.utils.data.DataLoader): data loader
    - device (torch.device): cuda or cpu
    - task (str): train, val, test
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
    probs = []
    preds = []
    tics = []
    secs = []
    total = 0
    if task in ["val", "test"]:
        model.eval()
    elif task == "train":
        model.train()
    else:
        raise NameError("Only train, val or test is allowed as task")
    
    with trange(len(data_loader)) as t:
        for i, batch in enumerate(data_loader):
            # unpack batch from dataloader
            (x, tic, sec), y = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            y_bin = (y > 0.5).float()
            print(probs.shape)
            print(probs)
            print(preds)
            print(y.shape)

            # compute loss on logits
            loss = criterion(logits, torch.unsqueeze(y, 1))
            avg_loss.update(loss.data.cpu().item(), x.size(0))     
            
            if task == "train":
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            probs = probs.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            y_bin = y_bin.detach().cpu().numpy()
            # collect the model outputs
            targets += y_bin.tolist()
            targets_bin += y_bin.tolist()
            probs += probs.tolist()
            preds += preds.tolist()
            tics += tic.tolist()
            secs += sec.tolist()
            total += logits.size(0)

            t.update()

    acc = accuracy_score(targets_bin, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(targets_bin, preds)
    auc = roc_auc_score(targets_bin, probs)

    if task == "test":
        return avg_loss.avg, acc, f1, prec, rec, auc, probs, targets, tics, secs, total
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
                task="val")
    print(f"\ninitial loss: {best_loss}, acc: {best_acc}")
    best_epoch = 0
    try:
        # Training loop
        for epoch in range(args.epochs):
            
            train_loss, train_acc, train_f1, train_prec, train_rec, train_auc = evaluate(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                data_loader=train_loader,
                device=args.device,
                task="train")

            # log
            # TODO track lr etc as well if using scheduler

            # evaluate on val set
            val_loss, val_acc, val_f1, val_prec, val_rec, val_auc = evaluate(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                data_loader=train_loader,
                device=args.device,
                task="val")

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

            print(
                f"\Epoch {epoch+1}/{args.epochs}: \ntrain/loss: {train_loss}, train/acc: {train_acc}, train/prec: {train_prec} train/rec: {train_rec}, train/f1: {train_f1}, train/auc: {train_auc}"
                f"\nval/loss: {val_loss}, val/acc: {val_acc}, val/prec: {val_prec}, val/rec: {val_rec}, val/f1: {val_f1}, val/auc: {val_auc}"
            )

            # patience
            if (epoch - best_epoch > args.patience):
                print("\nEarly stopping")
                break
    
    except KeyboardInterrupt:
        pass

    return model
