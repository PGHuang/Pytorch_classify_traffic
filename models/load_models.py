import sys

sys.path.append("../")
import logging
from torchstat import stat

from models.resnet50 import resnet50


def load_net(cfg):
    c = cfg.model

    if c.arch == "resnet50":
        model = resnet50(**c.conf)
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
    print(oup.shape)

    stat(net, (3, 360, 640))
