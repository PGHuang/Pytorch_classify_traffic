import torch
import math


# 热启动的 Cos学习率调整
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing with warmup
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        start_factor (float): Start factor of warmup
        warmup_size (int): period of warmup.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, T_max, eta_min=1e-6, start_factor=0.1, warmup_size=10, last_epoch=-1):
        self.T_max = T_max - warmup_size
        self.eta_min = eta_min
        self.warmup_size = warmup_size
        assert start_factor < 1.0
        self.start_factor = start_factor
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_size:
            factor = self.start_factor + self.last_epoch * (1.0 - self.start_factor) / (self.warmup_size - 1)
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) * (
                    1 + math.cos(math.pi * (self.last_epoch - self.warmup_size) / self.T_max)) / 2
                    for base_lr in self.base_lrs]
