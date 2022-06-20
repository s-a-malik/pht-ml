"""train.py
Utility functions and classes for model training and evaluation.
"""

import time
from collections import defaultdict

from tqdm.autonotebook import trange

import wandb

import torch

import numpy as np

from sklearn.metrics import roc_auc_score

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
    - loss: average loss on data_loader
    - acc: accuracy on data_loader
    - f1: f1 score on data_loader
    - prec: precision on data_loader
    - rec: recall on data_loader
    in addition, if test:
    - auc (float): roc_auc_score on data_loader
    - results (dict):
        - fluxs (list): predicted fluxes
        - probs (list): predicted probabilities
        - targets (list): true fluxes
        - targets_bin (list): true binary targets
        - tics (list): tic ids
        - secs (list): secs
        - tic_injs (list): tic injections
        - snrs (list): snrs
        - durations (list): durations
        - periods (list): periods
        - depths (list): depths
        - eb_prim_depths (list): primary eb depth
        - eb_sec_depths (list): secondary eb depth
        - eb_periods (list): eb periods
        - classes (list): classes (eb, synth_planet, real)
    """
    avg_loss = utils.AverageMeter()
    true_negatives = 0
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    total = 0
    if (task == "test") or (save_examples != -1):
        results = defaultdict(list)
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

            flux = flux.detach().cpu().numpy()
            prob = prob.detach().cpu().numpy()
            prob = np.squeeze(prob)
            pred = pred.detach().cpu().numpy()
            pred = np.squeeze(pred)
            y_bin = y_bin.detach().cpu().numpy()

            # sum up fp, tp
            true_positives += np.sum(pred * y_bin)
            false_positives += np.sum(pred * (1 - y_bin))
            false_negatives += np.sum((1 - pred) * y_bin)
            true_negatives += np.sum((1 - pred) * (1 - y_bin))
            total += y_bin.shape[0]

            # collect the model outputs
            if ((task == "test") or (save_examples != -1)):  
                # only save first few batches if just plotting
                if (save_examples != -1) and (len(results["targets"]) > 5000):
                    pass
                # else collect all test results
                else:      
                    results["targets"] += y.tolist()
                    results["targets_bin"] += y_bin.tolist()
                    results["probs"] += prob.tolist()
                    results["preds"] += pred.tolist()
                    results["tics"] += x["tic"].tolist()
                    results["secs"] += x["sec"].tolist()
                    results["tic_injs"] += x["tic_inj"].tolist()
                    results["snrs"] += x["snr"].tolist()
                    results["durations"] += x["duration"].tolist()
                    results["periods"] += x["period"].tolist()
                    results["depths"] += x["depth"].tolist()
                    results["eb_prim_depths"] += x["eb_prim_depth"].tolist()
                    results["eb_sec_depths"] += x["eb_sec_depth"].tolist()
                    results["eb_periods"] += x["eb_period"].tolist()
                    results["classes"] += x["class"]
                    results["tois"] += x["toi"].tolist()
                    results["ctcs"] += x["ctc"].tolist()
                    results["ctois"] += x["ctoi"].tolist()
                    results["tces"] += x["tce"].tolist()
                    results["tic_noises"] += x["tic_noise"].tolist()
                    # save flux only if plotting
                    if save_examples != -1:
                        results["fluxs"] += flux.tolist()

            t.update()

            # profiler.step()

    # compute metrics manually, handling zero division. 
    # TODO this could be done in a simpler way
    acc = np.divide((true_positives + true_negatives), total,  out=np.zeros_like((true_positives + true_negatives)), where=total!=0)
    prec = np.divide(true_positives, (true_positives + false_positives),  out=np.zeros_like(true_positives), where=(true_positives + false_positives)!=0)
    rec = np.divide(true_positives, (true_positives + false_negatives),  out=np.zeros_like(true_positives), where=(true_positives + false_negatives)!=0)
    f1 = np.divide(2 * prec * rec, (prec + rec), out=np.zeros_like(prec), where=(prec + rec)!=0)

    # save example predictions to wandb for inspection
    if save_examples != -1:
        print("saving example predictions")
        utils.save_examples(results, save_examples)

    if task == "test":
        auc = roc_auc_score(results["targets_bin"], results["probs"])
        return avg_loss.avg, acc, f1, prec, rec, auc, results
    else:
        return avg_loss.avg, acc, f1, prec, rec


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
    best_loss, best_acc, _, _, _ = evaluate(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                data_loader=val_loader,
                device=args.device,
                task="val",
                save_examples=-1 if args.example_save_freq == -1 else 0)
    print(f"\ninitial loss: {best_loss}, acc: {best_acc}")
    # save initial checkpoint
    checkpoint_dict = {
        "epoch": 0,
        "state_dict": model.state_dict(),
        "best_loss": best_loss,
        "optimizer": optimizer.state_dict(),
        "args": vars(args)
    }
    utils.save_checkpoint(checkpoint_dict, is_best=True)
    best_epoch = 0
    start_time = time.time()
    try:
        # Training loop
        for epoch in range(1, args.epochs):
            epoch_start = time.time()
            train_loss, train_acc, train_f1, train_prec, train_rec = evaluate(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                data_loader=train_loader,
                device=args.device,
                task="train",
                save_examples=-1)

            # evaluate on val set
            if (args.example_save_freq != -1) and ((epoch) % args.example_save_freq == 0):
                save_examples = epoch
            else:
                save_examples = -1
            val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(
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
                    "train/loss": train_loss,
                    "val/acc": val_acc,
                    "val/f1": val_f1,
                    "val/prec": val_prec,
                    "val/rec": val_rec,
                    "val/loss": val_loss,
                    "epoch": epoch,
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
                f"\nEpoch {epoch+1}/{args.epochs}: \ntrain/loss: {train_loss}, train/acc: {train_acc}, train/prec: {train_prec} train/rec: {train_rec}, train/f1: {train_f1}"
                f"\nval/loss: {val_loss}, val/acc: {val_acc}, val/prec: {val_prec}, val/rec: {val_rec}, val/f1: {val_f1}"
            )

            # patience
            if (epoch - best_epoch > args.patience):
                print("\nEarly stopping...")
                break
    
    except KeyboardInterrupt:
        pass

    return model, epoch + 1


def init_model(args):
    """Initialize model
    """
    model_name = f"{args.model}_{args.bin_factor}" if args.model != "dense" else args.model

    if model_name == "dense":
        model = nets.SimpleNetwork(
            input_dim=int(SHORTEST_LC / args.bin_factor),
            hid_dims=args.hid_dims,
            output_dim=1,
            non_linearity=args.activation,
            dropout=args.dropout
        )
    elif model_name == "ramjet_3":
        model = nets.RamjetBin3(
            input_dim=int(SHORTEST_LC / args.bin_factor),
            output_dim=1,
            dropout=0.1
        )
    elif model_name == "ramjet_7":
        model = nets.RamjetBin7(
            input_dim=int(SHORTEST_LC / args.bin_factor),
            output_dim=1,
            dropout=0.1
        )
    elif model_name == "resnet_7":
        model = nets.ResNetBin7(
            input_dim=int(SHORTEST_LC / args.bin_factor),
            output_dim=1,
            dropout=0.1
        )
    elif model_name == "resnet_big_7":
        model = nets.ResNetBigBin7(
            input_dim=int(SHORTEST_LC / args.bin_factor),
            output_dim=1,
            dropout=0.1
        )
    else:
        raise NameError(f"Unknown model {args.model}")
    
    # for tracking gradients etc.
    wandb.watch(model, log="all")  
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
