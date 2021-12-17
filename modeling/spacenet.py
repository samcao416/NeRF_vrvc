#The MLP Network
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import time

from utils import Trigonometric_kernel

class SpaceNet(nn.Module):


    def __init__(self, cfg):
        super(SpaceNet, self).__init__()

        self.use_dir = cfg.MODEL.USE_DIR

        self.tri_kernel_pos = Trigonometric_kernel(L=10, include_input = cfg.MODEL.TKERNEL_INC_RAW)
        if self.use_dir:
            self.tri_kernel_dir = Trigonometric_kernel(L=4, include_input = cfg.MODEL.TKERNEL_INC_RAW)

        self.pos_dim = self.tri_kernel_pos.calc_dim(3)
        if self.use_dir:
            self.dir_dim = self.tri_kernel_dir.calc_dim(3)
        else:
            self.dir_dim = 0

        backbone_dim = cfg.MODEL.BACKBONE_DIM
        head_dim = int(backbone_dim / 2)

        # 4-layer MLP for density feature
        self.stage1 = nn.Sequential(
                    nn.Linear(self.pos_dim, backbone_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_dim, backbone_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_dim, backbone_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_dim, backbone_dim),
                    nn.ReLU(inplace=True),
                )
        # 4-layer MLP for density feature with a skipped input and stage1 output
        self.stage2 = nn.Sequential(
                    nn.Linear(backbone_dim+self.pos_dim, backbone_dim), #?: Seems different from the structure in the paper
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_dim, backbone_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_dim, backbone_dim),
                    nn.ReLU(inplace=True),
                )
        # 1-layer MLP for density
        self.density_net = nn.Sequential(
                    nn.Linear(backbone_dim, 1),
                    # density value should be more than zero
                    # nn.ReLU(inplace=True)
                )
        # 2-layer MLP for rgb
        self.rgb_net = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_dim+self.dir_dim, head_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(head_dim, 3)
                )


    '''
    INPUT
    pos: 3D positions (N,L,c_pos) or (N,c_pos)
    rays: corresponding rays  (N,6)

    OUTPUT

    rgb: color (N,L,3) or (N,3)
    density: (N,L,1) or (N,1)

    N is the number of rays

    '''
    def forward(self, pos, rays):

        rgbs = None
        if rays is not None and self.use_dir:

            dirs = rays[...,3:6] 
            # Normalization for a better performance when training, may not necessary
            dirs = dirs/torch.norm(dirs,dim=-1,keepdim = True)

        # When input is [N, L, 3], it will be set to True
        bins_mode = False
        if len(pos.size())>2:
            bins_mode = True
            L = pos.size(1)
            pos = pos.reshape((-1,3))     # (N, 3)
            if rays is not None and self.use_dir:
                dirs = dirs.unsqueeze(1).repeat(1,L,1)
                dirs = dirs.reshape((-1,3))   # (N, 3)
        
        # Positional encoding
        pos = self.tri_kernel_pos(pos)
        if dirs is not None:
            dirs = self.tri_kernel_dir(dirs)
        # 8-layer MLP for density
        x = self.stage1(pos)
        x = self.stage2(torch.cat([x,pos], dim=1))
        density = self.density_net(x)
        # MLP for rgb with or without direction of ray
        x1 = 0
        if rays is not None and self.use_dir:
            x1 = torch.cat([x, dirs], dim=1)
        else:
            x1 = x.clone()

        rgbs = self.rgb_net(x1)

        if bins_mode:
            density = density.reshape((-1,L,1))
            if rays is not None:
                rgbs = rgbs.reshape((-1, L, 3))

        return rgbs, density



         


        


        

        


