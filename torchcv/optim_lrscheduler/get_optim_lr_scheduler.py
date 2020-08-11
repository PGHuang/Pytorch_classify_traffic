import sys
import logging
from torch import optim
from .warmup_cosine import WarmupCosineLR


def get_optimizer(net, init_lr, cfg):
    c = cfg.optimizer

    if c.type == "Adam":
        optimizer = optim.Adam(net, init_lr, **c.conf)
    elif c.type == "SGD":
        optimizer = optim.SGD(net, init_lr, **c.conf)
    elif c.type == "Adadelta":
        optimizer = optim.Adadelta(net, init_lr, **c.conf)
    else:
        logging.critical(f"Optimizer type {c.type} not defined!")
        sys.exit()

    return optimizer


def get_lr_scheduler(optimizer, cfg):
    c = cfg.lr_scheduler
    
    if c.type == "WarmupCosineLR":
        lr_scheduler = WarmupCosineLR(optimizer, **c.conf)
    elif c.type == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **c.conf)
    elif c.type == "MultiStepLR":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **c.conf)
    else:
        logging.critical(f"LR_scheduler type {c.type} not defined!")
        sys.exit()

    return lr_scheduler
