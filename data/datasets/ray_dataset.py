import torch
import numpy as np
from math import sin, cos, pi
import os
from PIL import Image
import torchvision
import random
from .frame_dataset import SynImageDataset
from utils import ray_sampling_syn, generate_rays_syn, ray_sampling_mip, generate_rays_mip

class Syn_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg):

        super(Syn_Dataset, self).__init__()

        #Save input
        self.dataset_path = cfg.DATASETS.TRAIN
        self.tmp_rays = cfg.DATASETS.TMP_RAYS
        self.output_dir = cfg.OUTPUT_DIR

        #get data from datasets
        self.image_dataset = SynImageDataset(cfg) #get into frame_dataset.py
        self.angles = self.image_dataset.angles
        self.camera_num = self.image_dataset.cam_num #cam_num is the total number of images

        self.rays = []
        self.colors = []
        # TODO
        self.bbox = []

        #Check if we have already generated rays
        tmp_ray_path = os.path.join(self.output_dir, self.tmp_rays)

        if not os.path.exists(tmp_ray_path):
            print('There is no ray generated before, start generating rays...')
            os.makedirs(tmp_ray_path)

        #generating rays
        if not os.path.exists(os.path.join(tmp_ray_path, 'near_fars.pt')) or cfg.clean_ray:
            print('generating rays...')
            rays_tmp = []
            colors_tmp = []
            near_fars_tmp = []
            viewdirs_tmp = []
            radii_tmp = []
            lossmult_tmp = []

            for i in range(0, self.image_dataset.cam_num):
                print('Generating Image No.%d rays' % (i))
                
                image, poses, focal, mask, near_far = self.image_dataset.get_data(i) #poses = camear poses important line!
                if not mask:
                    print('Skipping Image %d by mask.' % (i))
                    continue
                
                rays, colors, radii, lossmult = ray_sampling_mip(image, poses, focal) #get into ray_sampling.py
                near_fars_tmp.append(near_far.repeat(rays.size(0), 1))
                rays_tmp.append(rays) #rays: H*W, 6
                colors_tmp.append(colors) #colors: H*W, 3
                radii_tmp.append(radii) #raddi: H*W, 1
                lossmult_tmp.append(lossmult) #lossmult: H*W, 1

            self.rays = torch.cat(rays_tmp, 0) # (N * H * W, 6)
            self.colors = torch.cat(colors_tmp, 0) #(N * H * W, 3)
            self.radii = torch.cat(radii_tmp, 0) #(N * H * W, 1)
            self.lossmult = torch.cat(lossmult_tmp, 0) #(N * H * W, 1)
            self.near_fars = torch.cat(near_fars_tmp, 0)

            torch.save(self.rays, os.path.join(tmp_ray_path, 'rays.pt'))
            torch.save(self.colors, os.path.join(tmp_ray_path, 'colors.pt'))
            torch.save(self.near_fars, os.path.join(tmp_ray_path, 'near_fars.pt'))
            torch.save(self.radii, os.path.join(tmp_ray_path, 'radii.pt'))
            torch.save(self.lossmult, os.path.join(tmp_ray_path, 'lossmult.pt'))
        else:
            print('There are rays generated before, loading rays...')
            self.rays = torch.load(os.path.join(tmp_ray_path, 'rays.pt'), map_location = 'cpu')
            self.colors = torch.load(os.path.join(tmp_ray_path, 'colors.pt'), map_location = 'cpu')
            self.near_fars = torch.load(os.path.join(tmp_ray_path, 'near_fars.pt'), map_location = 'cpu')
            self.radii = torch.load(os.path.join(tmp_ray_path, 'radii.pt'), map_location = 'cpu')
            self.lossmult = torch.load(os.path.join(tmp_ray_path, 'lossmult.pt'), map_location = 'cpu')
        

    
    #print('Generated %d rays' % self.rays.shape[0])

    def __len__(self):
        return self.rays.shape[0]

    def __getitem__(self, index):
        return self.rays[index, :], self.colors[index, :], self.near_fars[index, :], \
               self.radii[index, :], self.lossmult[index, :]
        
    def vis(self, cfg):
        pos = []
        color = []

        num = cfg.TEST.SAMPLE_NUMS
        step = cfg.TEST.STEP_SIZE
        step_num = cfg.TEST.STEP_NUM

        raysrgb = np.concatenate([self.rays, self.colors], axis=1)
        raysrgb = raysrgb[np.all(raysrgb[:,6:9] > 0.15, axis=1)]

        print(raysrgb.shape)

        np.random.shuffle(raysrgb)

        for i in range(step_num):
            pos.append(raysrgb[:num, :3] + i*step*raysrgb[:num, 3:6])
            color.append(raysrgb[:num, 6:9])
            
        pos = np.concatenate(pos, axis=0)
        color = np.concatenate(color, axis=0)
        pts_out = np.concatenate([pos, color], axis = 1)
        ply_dir = os.path.join(cfg.OUTPUT_DIR, 'pointclouds.txt')

        np.savetxt(ply_dir, pts_out)

        print("point clouds saved")
        exit()


class Syn_Dataset_View(torch.utils.data.Dataset):
    def __init__(self, cfg):

        super(Syn_Dataset_View, self).__init__()
        self.dataset_path = cfg.DATASETS.TRAIN
        
        self.dataset = SynImageDataset(cfg) #get into frame_dataset.py

        self.camera_num = self.dataset.cam_num

        # TODO
        self.box = []

        self.near_far = torch.Tensor([cfg.DATASETS.FIXED_NEAR,cfg.DATASETS.FIXED_FAR]).reshape(1,2)

    def __len__(self):
        return 1
    
    def get_fixed_image(self, index_view):

        image, pose, focal, _ , near_far = self.dataset.get_data(index_view)
        image = torch.Tensor(image)
        rays, colors, radii, lossmult = ray_sampling_mip(image, pose, focal)
        return rays, colors, image, near_far.repeat(rays.size(0), 1), radii, lossmult
    
    def __getitem__(self, index):

        index_view = np.random.randint(0, self.camera_num)
        print(index_view)
        return self.get_fixed_image(index_view)


class Syn_Dataset_Render(torch.utils.data.Dataset):

    def __init__(self, cfg):

        super(Syn_Dataset_Render, self).__init__()

        self.dataset = SynImageDataset(cfg) #get into frame_dataset.py
        self.camera_num = self.dataset.cam_num
        image, poses, focal, _ , near_far= self.dataset.get_data(0)
        self.image = torch.Tensor(image)
        self.H, self.W = self.image.shape[:2]
        self.focal = focal
        
        self.near_far = torch.Tensor([self.dataset.near_fars.min(), self.dataset.near_fars.max()]).reshape(1,2)

        # TODO
        self.bbox = []

    def get_gt(self, index):
        image, poses, focal, _, near_far = self.dataset.get_data(index)
        return image, poses, focal

    def get_rays(self, pose):
        rays  = generate_rays_mip(self.H, self.W, pose, self.focal)
        return rays, self.near_far.repeat(rays.size(0),1)

    def get_gt_rays(self, i):
        image, pose, focal, _ , near_far = self.dataset.get_data(i)
        return self.get_rays(pose)
