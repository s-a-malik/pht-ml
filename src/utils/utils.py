"""General utilities
"""

import os
import shutil

import wandb


from models import nets

SHORTEST_LC = 17546

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
    else:
        raise NameError(f"Unknown optimizer {args.optimizer}")
    
    if args.loss == "BCE":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.loss == "MSE":
        criterion = torch.nn.MSELoss()
    else:
        raise NameError(f"Unknown loss function {args.loss}")

    return optimizer, criterion


def init_model(args):
    """Initialize model
    """
    if args.model == "dense":
        model = nets.SimpleNetwork(
            input_dim=SHORTEST_LC,
            hid_dims=args.hid_dims,
            output_dim=1,
            non_linearity=args.activation,
            dropout=args.dropout
        )
    # if args.model == "resnet18":
    #     model = nets.ResNet18(args.num_classes)
    # elif args.model == "resnet34":
    #     model = nets.ResNet34(args.num_classes)
    # elif args.model == "resnet50":
    #     model = nets.ResNet50(args.num_classes)
    else:
        raise NameError(f"Unknown model {args.model}")
    model.to(args.device)


    return model

    pass

def load_checkpoint(model, optimizer, device, checkpoint_file: str):
    """Loads a model checkpoint.
    Params:
    - model (nn.Module): initialised model
    - optimizer (nn.optim): initialised optimizer
    - device (torch.device): device model is on
    Returns:
    - model with loaded state dict
    - optimizer with loaded state dict
    """
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Loaded {checkpoint_file}, "
          f"trained to epoch {checkpoint['epoch']+1} with best loss {checkpoint['best_loss']}")

    return model, optimizer

# save/load checkpoints
def save_checkpoint(checkpoint_dict: dict, is_best: bool):
    """Saves a model checkpoint to file. Keeps most recent and best model.
    Params:
    - checkpoint_dict (dict): dict containing all model state info, to pickle
    - is_best (bool): whether this checkpoint is the best seen so far.
    """
    checkpoint_file = os.path.join(wandb.run.dir, "ckpt.pth.tar")
    best_file = os.path.join(wandb.run.dir, "best.pth.tar")
    torch.save(checkpoint_dict, checkpoint_file)

    if is_best:
        shutil.copyfile(checkpoint_file, best_file)

