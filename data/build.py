# encoding: utf-8
"""
@author:  Minye Wu
@GITHUB: wuminye
"""

from torch.utils import data
import numpy as np
from .datasets.ray_dataset import Syn_Dataset, Syn_Dataset_View, Syn_Dataset_Render#, Indoor_Dataset, Indoor_Dataset_View, Indoor_Dataset_Render
from .transforms import build_transforms


def make_ray_data_loader(cfg, is_train=True): #TODO

    batch_size = cfg.SOLVER.IMS_PER_BATCH

    if cfg.DATASETS.TYPE == 'syn':
        datasets = Syn_Dataset(cfg)
    else:
        print('Error: undefined dataset type: ' + cfg.DATASETS.TYPE)
    #Then get into ray_dataset.py

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return data_loader, datasets

def make_ray_data_loader_view(cfg, is_train=False):

    batch_size = cfg.SOLVER.IMS_PER_BATCH

    if cfg.DATASETS.TYPE == 'syn':
        datasets = Syn_Dataset_View(cfg)

    else:
        print('Error: undefined dataset type: ' + cfg.DATASETS.TYPE)
    #Then get into ray_dataset.py

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return data_loader, datasets

def make_ray_data_loader_render(cfg, is_train=False):

    batch_size = cfg.SOLVER.IMS_PER_BATCH 
        

    #datasets = CryoET_Dataset_Render(cfg)
    if cfg.DATASETS.TYPE == 'syn':
        datasets = Syn_Dataset_Render(cfg)

    else:
        print('Error: undefined dataset type: ' + cfg.DATASETS.TYPE)
    #Then get into ray_dataset.py

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return data_loader, datasets