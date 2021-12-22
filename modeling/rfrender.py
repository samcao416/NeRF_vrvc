
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import math

from utils import Trigonometric_kernel, sample_pdf
from layers.RaySamplePoint import RaySamplePoint, RaySamplePoint_Near_Far, RaySamplePoint_Mip, ResamplePointMip
from .spacenet import SpaceNet

from layers.render_layer import VolumeRenderer, VolumeRendererMip, gen_weight, Projector, AlphaBlender
from layers.camera_transform import CameraTransformer, TiltRefiner
import time

import copy

import pdb
class RFRender(nn.Module):

    def __init__(self, cfg, camera_num): #camera_num is the number of images
        super(RFRender, self).__init__()

        self.coarse_ray_sample = cfg.MODEL.COARSE_RAY_SAMPLING
        self.fine_ray_sample = cfg.MODEL.FINE_RAY_SAMPLING
        self.sample_method = cfg.MODEL.SAMPLE_METHOD

        # Related to input of networks
        self.use_dir = cfg.MODEL.USE_DIR
        self.camera_num = camera_num

        #Pose refinement part
        self.cam_pose=CameraTransformer(self.camera_num, True)

       
        # Ray Sampling Methods
        if cfg.MODEL.SAMPLE_METHOD == 'NEAR_FAR':
            self.rsp_coarse = RaySamplePoint_Mip(sample_num = self.coarse_ray_sample)   # use near far to sample points on rays
            self.rsp_fine   = ResamplePointMip(sample_num=self.fine_ray_sample)
        #elif cfg.MODEL.SAMPLE_METHOD == 'BBOX':
        #    self.rsp_coarse = RaySamplePoint(self.coarse_ray_sample)            # use bounding box to define point sampling ranges on rays

        # NeRF Network 
        self.spacenet = SpaceNet(cfg)
        
        # Whether use the same network
        if cfg.MODEL.SAME_SPACENET:
            self.spacenet_fine = self.spacenet
        else:
            self.spacenet_fine = copy.deepcopy(self.spacenet)

        # Volume Rendering Method 
        if cfg.MODEL.BLENDING_SCHEME == "VOLUME RENDERING":
            #self.volume_render = VolumeRenderer(boarder_weight = cfg.MODEL.BOARDER_WEIGHT)
            self.volume_render = VolumeRendererMip()
        self.maxs = None
        self.mins = None

    '''
    INPUT

    rays: rays  (N,6)
    bboxes: bounding boxes (N,L,8,3)

    OUTPUT

    rgbs: color of each ray (N,3) 
    depths:  depth of each ray (N,1) 

    '''
    def forward(self, rays, radii, lossmult, bboxes=None, near_far=None, rendering=False):
        
        results = {} 

        # [x,y,z,dx,dy,dz]
        ray_size = 6

        K = rays.size(0)
        ray_mask = None 

        if self.sample_method == 'NEAR_FAR':
            assert near_far is not None, 'require near_far as input '
            sampled_rays_coarse_t, sampled_rays_coarse_xyz  = self.rsp_coarse.forward(rays, radii, near_far = near_far)
            rays_t = rays

        if rays_t.size(0) > 1:
        
            # Sampling Number
            L1 = self.coarse_ray_sample
            L2 = self.fine_ray_sample

            # Detach if we do not need to refine camera pose
            sampled_rays_coarse_t = sampled_rays_coarse_t.detach() # N, L + 1
            sampled_rays_coarse_xyz = sampled_rays_coarse_xyz.detach() # N, L, 3
            
            # Canonical NeRF 
            colors, density = self.spacenet(sampled_rays_coarse_xyz, rays_t )

            # Volume Rendering
            color_0, depth_0, acc_0, weights_0 = self.volume_render(density, colors, sampled_rays_coarse_t, rays_t[:, 3:6])

            '''
            # Set point density behind image plane into zero
            density[sampled_rays_coarse_t[:,:,0]<0,:] = 0.0

            
            
            # Importance Sampling
            z_samples = sample_pdf(sampled_rays_coarse_t.squeeze(), weights_0.squeeze()[...,1:-1], N_samples = self.fine_ray_sample)
            z_samples = z_samples.detach()   # (N,L)

            # Sorting
            z_vals_fine, _ = torch.sort(torch.cat([sampled_rays_coarse_t.squeeze(), z_samples], -1), -1) #(N, L1+L2)
            samples_fine_xyz = z_vals_fine.unsqueeze(-1)*rays_t[:,3:6].unsqueeze(1) + rays_t[:,:3].unsqueeze(1)  # (N,L1+L2,3)

            # Detach if we do not need to refine camera pose
            samples_fine_xyz = samples_fine_xyz.detach()
            z_vals_fine = z_vals_fine.detach()
            '''
            sampled_rays_fine_t, sampled_rays_fine_xyz = self.rsp_fine.forward(rays_t, radii, weights_0, sampled_rays_coarse_t)
            # Canonical NeRF 
            colors, density = self.spacenet_fine(sampled_rays_fine_xyz, rays_t)
            #print("colors shape: ", colors.shape) #[Batch_size, L1 + L2, 3]

            # Set point density behind image plane into zero
            #density[z_vals_fine < 0] = 0

            # Volume Rendering
            color, depth, acc, weights = self.volume_render(density, colors, sampled_rays_fine_t, rays_t[:, 3:6])
            
            C = color.shape[-1]
        
            if self.sample_method == 'BBOX':
                color_c = torch.zeros(K,C,device = rays.device)
                color_c[ray_mask] = color_0

                depth_c = torch.zeros(K,1,device = rays.device)
                depth_c[ray_mask] = depth_0

                acc_c = torch.zeros(K,1,device = rays.device)
                acc_c[ray_mask] = acc_0
            else:
                color_c = color_0
                depth_c = depth_0
                acc_c = acc_0
            
            results['coarse_color'] = color_c
            results['coarse_depth'] = depth_c
            results['coarse_acc'] = acc_c

            if self.sample_method == 'BBOX':
                color_f = torch.zeros(K,C,device = rays.device)
                color_f[ray_mask] = color

                depth_f = torch.zeros(K,1,device = rays.device)
                depth_f[ray_mask] = depth

                acc_f = torch.zeros(K,1,device = rays.device)
                acc_f[ray_mask] = acc

                results['ray_mask'] = ray_mask
            else:
                color_f = color
                depth_f = depth
                acc_f = acc
            
            results['fine_color'] = color_f
            results['fine_depth'] = depth_f
            results['fine_acc'] = acc_f
            results['ray_mask'] = torch.ones_like(color_f).type(torch.bool)
        else:
            C = self.cfg.DATASETS.COLOR_CHANNEL
            results['coarse_color'] =  torch.zeros(K,C,device = rays.device)
            results['coarse_depth'] = torch.zeros(K,1,device = rays.device)
            results['coarse_acc'] = torch.zeros(K,1,device = rays.device)
            results['ray_mask'] = ray_mask
            results['fine_color'] = torch.zeros(K,C,device = rays.device)
            results['fine_depth'] = torch.zeros(K,1,device = rays.device)
            results['fine_acc'] = torch.zeros(K,1,device = rays.device)

        return results

    def set_coarse_sample_point(self, a):
        self.coarse_ray_sample = a
        self.rsp_coarse.set_coarse_sample_point(a)

    def set_fine_sample_point(self, a):
        self.fine_ray_sample = a

    def get_density(self, pos):
        _, density = self.spacenet_fine(pos, None)

        return density
