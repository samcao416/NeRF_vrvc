#The MLP Network
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import time

from utils import Trigonometric_kernel, integrated_pos_enc, pos_enc

class SpaceNet(nn.Module):


    def __init__(self, cfg):
        super(SpaceNet, self).__init__()

        self.use_dir = cfg.MODEL.USE_DIR

        #self.tri_kernel_pos = Trigonometric_kernel(L=10, include_input = cfg.MODEL.TKERNEL_INC_RAW)
        #if self.use_dir:
        #    self.tri_kernel_dir = Trigonometric_kernel(L=4, include_input = cfg.MODEL.TKERNEL_INC_RAW)
#
        #self.pos_dim = self.tri_kernel_pos.calc_dim(3)
        #if self.use_dir:
        #    self.dir_dim = self.tri_kernel_dir.calc_dim(3)
        #else:
        #    self.dir_dim = 0

        backbone_dim = cfg.MODEL.BACKBONE_DIM
        head_dim = int(backbone_dim / 2)

        self.min_deg_point = cfg.MODEL.MIP_MIN_DEG_POINT
        self.max_deg_point = cfg.MODEL.MIP_MAX_DEG_POINT
        self.deg_view = cfg.MODEL.DEG_VIEW

        # 4-layer MLP for density feature
        self.stage1 = nn.Sequential(
                    #nn.Linear(self.pos_dim, backbone_dim),
                    nn.LazyLinear(backbone_dim),
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
                    #nn.Linear(backbone_dim+self.pos_dim, backbone_dim),
                    nn.LazyLinear(backbone_dim),
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
                    #nn.Linear(backbone_dim+self.dir_dim, head_dim),
                    nn.LazyLinear(head_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(head_dim, 3)
                )

        #self.condition_net = nn.Sequential(
        #                nn.ReLU(inplace = True),
        #                nn.Linear(backbone_dim, 128)
        #)


    '''
    INPUT
    pos: 3D positions (N,L,3)
    rays: corresponding rays  (N,6)

    OUTPUT

    rgb: color (N,L,3) or (N,3)
    density: (N,L,1) or (N,1)

    N is the number of rays

    '''
    def forward(self, pos, rays):

        rgbs = None
        L = pos[0].shape[1] #sample_num
        if rays is not None and self.use_dir:

            dirs = rays[...,3:6] 
            # Normalization for a better performance when training, may not necessary
            dirs = dirs/torch.norm(dirs,dim=-1,keepdim = True)
            dirs = dirs.unsqueeze(1).repeat(1, L, 1)
            dirs = dirs.reshape([-1, 3])

        
        # Positional encoding
        pos = integrated_pos_enc(pos, self.min_deg_point, self.max_deg_point) # N, L, (max - min) * 6
        pos_dim = pos.shape[2]
        pos = pos.reshape(-1, pos_dim)
        dirs = pos_enc(dirs, min_deg = 0, max_deg = self.deg_view, append_identity = True) # N*L , 3 + (max-min)*6
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

        density = density.reshape((-1,L,1))
        if rays is not None:
            rgbs = rgbs.reshape((-1, L, 3))

        return rgbs, density



         


        


        

        


