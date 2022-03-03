import torch
import numpy as np
import os
from PIL import Image
import json



class SynImageDataset(torch.utils.data.Dataset):
    
    def __init__(self, cfg):

        super(SynImageDataset, self).__init__()

        dataset_path = cfg.DATASETS.TRAIN

        #generate the json path
        self.json_path = os.path.join(dataset_path, 'transforms_train.json')
        f = open(self.json_path)
        json_file = json.load(f)
        self.angles = json_file['camera_angle_x']
        self.cam_num = len(json_file['frames'])
        
        #load images and poses
        self.images = []
        self.poses = []
        self.exts = []
        self.near_fars = []
        self.mask = None
        for i in range(self.cam_num):
            #generate the image file path
            image_path = os.path.join(dataset_path,json_file['frames'][i]['file_path']) + '.png'
            image = Image.open(image_path)
            if cfg.DATASETS.FACTOR != 1:
                image = image.resize((image.width // cfg.DATASETS.FACTOR, image.height // cfg.DATASETS.FACTOR))
            image = np.array(image)
            image = image[...,0:3] #drop the alpha channel of images
            
            self.images.append(image)

            #load poses
            pose = json_file['frames'][i]['transform_matrix']
            # pose[0][2] = -pose[0][2]
            # pose[1][2] = -pose[1][2]
            # pose[2][2] = -pose[2][2]
            self.poses.append(pose)
            self.exts.append(pose)

        self.images = np.array(self.images)
        self.images = (self.images / 255.0).astype(np.float32)
        H, W = self.images[0].shape[:2]
        self.focal = 0.5 * W / np.tan(0.5 * self.angles)
        self.poses = torch.Tensor(self.poses)
        self.exts = torch.inverse(torch.Tensor(self.exts))
        
        #get near_fars for each image
        self.vs = self.poses[:,0:3,3] 
        vs = self.vs.clone().unsqueeze(-1) # (N, 3, 1)
        vs = torch.cat([vs, torch.ones(vs.size(0), 1, vs.size(2))], dim = 1) #(N, 4, 1)
        pts = torch.matmul(self.exts.unsqueeze(1), vs) #(N, N, 4, 1) 
        pts_max = torch.max(pts, dim = 1)[0].squeeze() #(N, 4)
        pts_min = torch.min(pts, dim = 1)[0].squeeze() #(N, 4)
        pts_max = pts_max[:, 2].reshape(-1, 1) #(M, 1)
        pts_min = pts_min[:, 2].reshape(-1, 1) #(M, 1)
        
        near = pts_min * 0.5
        near[near < (pts_max * 0.1)] = pts_max[near < (pts_max * 0.1)] * 0.1
        far = pts_max * 2
        self.near_fars = torch.cat([near, far], dim = 1)
        # Use fixed near far if needed
        if cfg.DATASETS.FIXED_NEAR != -1:
            self.near_fars[:,0] = cfg.DATASETS.FIXED_NEAR
        if cfg.DATASETS.FIXED_FAR != -1:
            self.near_fars[:,1] = cfg.DATASETS.FIXED_FAR


    def __len__(self):
        return self.cam_num

    def get_data(self, image_id):
        if self.mask is not None:
            return self.images[image_id], self.poses[image_id], self.focal, self.mask[image_id], self.near_far[image_id]
        else:
            return self.images[image_id], self.poses[image_id], self.focal, 1, self.near_fars[image_id]
