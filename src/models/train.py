"""train.py
Utility functions and classes for model training and evaluation.
"""

import os

from tqdm.autonotebook import trange

import wandb

import torch

import numpy as np

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
    if task == "test":
        self.eval()
        test_targets = []
        test_pred = []
        test_tics = []
        test_secs = []
        test_total = 0
    else:
        loss_meter = utils.AverageMeter()
        if task == "val":
            self.eval()
        elif task == "train":
            self.train()
        else:
            raise NameError("Only train, val or test is allowed as task")

    with trange(len(train_dataloader)) as t:
        for i, batch in enumerate(data_loader):
            # unpack batch from dataloader
            (x, tic, sec), y = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.sigmoid(logits)
            print(preds.shape)
            print(y.shape)

            

            if task == "test":
                # collect the model outputs
                test_targets += y.tolist()
                test_pred += preds.tolist()
                test_tics += tic.tolist()
                test_secs += sec.tolist()
                test_total += logits.size(0)
            else:
                # compute loss on logits
                loss = criterion(logits, y)
                loss_meter.update(loss.data.cpu().item(), x.size(0))

                if task == "train":
                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            t.update()

    if task == "test":
        return test_pred, test_targets, test_tics, test_secs, test_total
    else:
        return loss_meter.avg


def training_run(args, model, optimizer, criterion, train_loader, val_loader):
    """Run training loop
    Returns:
    - model (nn.Module): trained model
    """
    # get best val loss
    best_loss, best_acc, _, _, _, _, _, _, _, _, _ = test_loop(
        args, model, val_loader)
    print(f"\ninitial loss: {best_loss}, acc: {best_acc}")
    best_epoch = 0
    try:
        # Training loop
        for epoch in range(args.epochs):
            
            train_loss, train_acc, train_f1, train_prec, train_rec = evaluate(
                batch=batch,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                task="train")

            # log
            # TODO track lr etc as well if using scheduler
            wandb.log(
                {
                    "train/acc": train_acc,
                    "train/f1": train_f1,
                    "train/prec": train_prec,
                    "train/rec": train_rec,
                    "train/loss": train_loss,
                })

            # evaluate on val set
            val_loss, val_acc, val_f1, val_prec, val_rec, val_lamda, _, _, _, _, _ = test_loop(
                args, model, val_loader, max_test_batches)
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                best_epoch = epoch
            wandb.log(
                {
                    "val/acc": val_acc,
                    "val/f1": val_f1,
                    "val/prec": val_prec,
                    "val/rec": val_rec,
                    "val/loss": val_loss,
                    "val/avg_lamda": val_lamda
                },
                step=batch_idx)

            # save checkpoint
            checkpoint_dict = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": opt.state_dict(),
                "args": vars(args)
            }
            utils.save_checkpoint(checkpoint_dict, is_best)

            print(
                f"\Epoch {epoch+1}/{args.epochs}: \ntrain/loss: {train_loss}, train/acc: {train_acc}"
                f"\nval/loss: {val_loss}, val/acc: {val_acc}"
            )

            # patience
            if (epoch - best_epoch > args.patience):
                print("\nEarly stopping")
                break
    
    except KeyboardInterrupt:
        pass

    return model


def test_run(args, model, criterion, test_loader):
    """Evaluate model on val/test set.
    Params:
    - args (argparse.Namespace): command line arguments
    - model (nn.Module): model to evaluate
    - criterion: loss function
    - test_loader (torch.utils.data.DataLoader): test loader
    Returns:
    - avg_test_acc (float): average test accuracy per task
    - avg_test_f1 (float): average test f1 per task
    - avg_test_prec (float): average test prec per task
    - avg_test_rec (float): average test rec per task
    - avg_test_loss (float): average test loss per task
    - test_preds (list(float)): list of class predictions
    - test_trues (list(float)): list of true labels
    - test_tics (list(int)): list of tic_ids of the test set
    - test_secs (list(int)): list of sectors of the test set
    """
    # TODO need to fix number of tasks/episodes etc. depending on batch, num_ways etc.

    avg_test_acc = AverageMeter()
    avg_test_f1 = AverageMeter()
    avg_test_prec = AverageMeter()
    avg_test_rec = AverageMeter()
    avg_test_loss = AverageMeter()
    test_preds = []
    test_trues = []
    query_idx = []
    support_idx = []
    support_lamdas = []
    avg_lamda = AverageMeter()

    for batch_idx, batch in enumerate(
            tqdm(test_dataloader,
                 total=max_num_batches,
                 position=0,
                 leave=True)):
        with torch.no_grad():
            test_loss, test_acc, test_f1, test_prec, test_rec, lamda, preds, trues, query, support, support_lamda = model.evaluate(
                batch=batch,
                optimizer=None,
                scheduler=None,
                num_ways=args.num_ways,
                device=args.device,
                task="test")

        avg_test_acc.update(test_acc)
        avg_test_f1.update(test_f1)
        avg_test_prec.update(test_prec)
        avg_test_rec.update(test_rec)
        avg_test_loss.update(test_loss)

        avg_lamda.update(lamda)
        test_preds += preds.tolist()
        test_trues += trues.tolist()
        query_idx += query.tolist()
        support_idx += support.tolist()
        support_lamdas += support_lamda.tolist()

        if batch_idx > max_num_batches - 1:
            break

    return avg_test_loss.avg, avg_test_acc.avg, avg_test_f1.avg, avg_test_prec.avg, avg_test_rec.avg, avg_lamda.avg, test_preds, test_trues, query_idx, support_idx, support_lamdas
