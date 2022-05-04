"""General utilities
"""

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


def init_optim(args):
    """Initialize optimizer
    """
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def init_model(args):
    """Initialize model
    """
    model.to(device)
    return model


def load_checkpoint(model, optimizer, scheduler, device, checkpoint_file: str):
    """Loads a model checkpoint.
    Params:
    - model (nn.Module): initialised model
    - optimizer (nn.optim): initialised optimizer
    - scheduler: initialised scheduler
    - device (torch.device): device model is on
    Returns:
    - model with loaded state dict
    - optimizer with loaded state dict
    - scheduler with loaded state dict
    """
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    print(f"Loaded {checkpoint_file}, "
          f"trained to epoch {checkpoint['epoch']+1} with best loss {checkpoint['best_loss']}")

    return model, optimizer, scheduler

