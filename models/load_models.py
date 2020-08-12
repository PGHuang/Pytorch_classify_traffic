import sys

sys.path.append("../")
import logging
from torchstat import stat

from models.resnet50 import resnet50
from models.resnet18_tv import Resnet18_tv


def load_net(cfg):
    c = cfg.model

    if c.arch == "resnet50":
        model = resnet50(**c.conf)
    elif c.arch == "resnet18_tv_pretrain":
        model = Resnet18_tv(**c.conf)
    elif c.arch == "resnet18_tv_woPretrain":
        model = Resnet18_tv(**c.conf)
    else:
        logging.critical(f"Model type {c.arch} not defined!")
        sys.exit()

    return model


if __name__ == '__main__':
    import torch
    from config import cfg

    net = load_net(cfg)
    inp = torch.randn(4, 3, 360, 640)
    oup = net(inp)

    stat(net, (3, 360, 640))
    print(oup.shape)
