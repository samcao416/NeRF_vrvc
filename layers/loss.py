import torch
import torch.nn as nn

def make_loss(cfg):
    if cfg.MODEL.LOSS == 'L2':
        return nn.MSELoss()
    elif cfg.MODEL.LOSS == 'L1':
        return nn.SmoothL1Loss()
