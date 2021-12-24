from config import cfg
import imageio
import os
import numpy as np
import torch
from data import make_ray_data_loader_render, get_iteration_path_and_iter
from modeling import build_model
from utils import batchify_ray
import math
import time
import copy

import matplotlib.pyplot as plt


# torch.cuda.set_device(args.gpu[0])
torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)


class NeuralRenderer:

    def __init__(self, cfg=None, device=None):
        self.images = []
        self.depths = []

        # Total image number rendered and saved in renderer
        self.image_num = 0

        self.fps = 30

        #Count for save multiple videos
        self.save_count = 0

        # All rendered poses and intrinsics aligned with images
        self.poses = []

        # auto saving dir
        self.dir_name = ''

        self.rendering_path = None
        self.rendering_count = 0

        ## Rendering area (lower bound of x, upper bound of x, lower bound of y, upper bound of y)
        self.area = [0, 1, 0, 1]

        if device is not None:
            torch.cuda.set_device(device)
            self.gpu = device

        if cfg is None:
            print('Warning: You created an empty render without config file of the model.')
            return
        else:
            self.cfg = cfg        

            # The dictionary save all rendered images and videos
            self.dataset_dir = self.cfg.OUTPUT_DIR
            self.output_dir = os.path.join(self.cfg.OUTPUT_DIR,'rendered')

            self.dataset, self.model = self.load_dataset_model()

            self.camera_num = self.dataset.camera_num

            # Image resolutions
            self.height = self.dataset.H
            self.width = self.dataset.W


    # load config file of the model
    def load(self, path):
        # Load cfg
        cfg.merge_from_file(path)
        self.cfg = cfg

        # The dictionary save all rendered images and videos
        self.dataset_dir = self.cfg.OUTPUT_DIR
        self.output_dir = os.path.join(self.cfg.OUTPUT_DIR,'rendered')

        self.dataset, self.model = self.load_dataset_model()

        if self.dataset is None:
            return False
        else:
            self.camera_num = self.dataset.camera_num
            # Image resolutions
            self.height = self.dataset.H
            self.width = self.dataset.W
            return True

    def load_dataset_model(self):
        
        _, dataset = make_ray_data_loader_render(cfg)
        model = build_model(cfg, dataset.camera_num).cuda()
        
        print(self.dataset_dir)
        para_file, iter0 = get_iteration_path_and_iter(self.dataset_dir)
            
        if not os.path.exists(para_file) or para_file == "":
            print('Error: The path of checkpoint does not exist.')
            return None, None
        print('Rendering from checkpoint: ',para_file)
        dict_0 = torch.load(os.path.join(self.dataset_dir,para_file))

        model.load_state_dict(dict_0['model'])

        return dataset, model

    def get_result(self, pose=None, camera_id=None):
        
        if pose != None:
            rays, near_far = self.dataset.get_rays(pose=pose)
        elif camera_id != None:
            rays, near_far = self.dataset.get_gt_rays(camera_id)

        rays = rays.cuda()
        # bbox = bbox.cuda()
        near_far = near_far.cuda()
        with torch.no_grad():
            results = batchify_ray(self.model, rays, near_far=near_far)

        return results

    # Get the points at the same plane by giving camera pose and plane depth
    def get_density(self, pose, depth):

        with torch.no_grad():
            pts = self.dataset.get_pts(pose, depth)
            pts = pts.cuda()
            
            results = {}

            density = self.model.get_density(pts)

            return density
    
    def get_phase(self, pose, depth):

        with torch.no_grad():

            pts = self.dataset.get_pts(pose, depth)
            rays,_,_ = self.dataset.get_rays(pose=pose)
            pts = pts.cuda()
            rays = rays.cuda()

            phase = self.model.get_phase(pts, rays) #TODO: No get_phase in rfrender.py
            
            return phase


    # Get ground truth image from dataset 
    def get_gt(self, camera_id):
        image, T = self.dataset.get_gt(camera_id)
        return image

    def render_gt(self):
        for i in range(self.camera_num):
            result = self.get_result(camera_id=i)

            color = self.color(result)
            depth = self.depth(result)

            self.images.append(color)
            self.depths.append(depth)

            self.save_color(color)
            self.save_depth(depth)

            self.image_num += 1
        
    #def render_phase(self, step_num, auto_save=True):
    #    
    #    # Central camera pose
    #    poses, _ = tilt_to_poses(0, 1, 1, up=[0, 1, 0])
    #    pose = poses[0]
#
    #    height = self.cfg.DATASETS.SAMPLE_HEIGHT
    #    up = height / 2
    #    down = -height / 2
#
    #    for i in range(step_num):
    #        print('Rendering image %d / %d' % (i, step_num))
    #        
    #        depth = up + (down-up) * i / step_num
#
    #        phase = self.get_phase(pose, depth)
    #        print(phase.shape)
    #        # normalize phase
    #        phase = torch.nn.functional.normalize(phase, dim=1)
    #        print(phase.shape)
    #        arg = torch.atan(phase[...,0] / phase[...,1])
    #        arg = arg.reshape(self.height, self.width, -1)
    #        # Normalize angles to 0 1
    #        arg = (arg + math.pi / 2) / math.pi
    #        arg = arg.cpu()
#
    #        # color = result['fine_color'].reshape(self.height,self.width,-1).cpu()
    #        # color = self.postprocessing(color)
#
    #        self.images.append(arg)
#
    #        if auto_save:
    #            self.save_color(arg)
#
    #        self.image_num += 1
#
#
    #    return

    # Save single color image
    def save_color(self, color, path = None):

        if path is None:
            if self.dir_name == '':
                save_dir = os.path.join(self.output_dir,'video_%d' % self.save_count,'image')
            else:
                save_dir = os.path.join(self.output_dir,self.dir_name+'_%d' % self.save_count,'image')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if not os.path.exists(os.path.join(save_dir,'color')):
                os.mkdir(os.path.join(save_dir,'color'))

            imageio.imwrite(os.path.join(save_dir,'color','%d.jpg'% self.image_num), color)
        else:
            imageio.imwrite(path, color)
        return

    # Save single depth image
    def save_depth(self, depth):
        if self.dir_name == '':
            save_dir = os.path.join(self.output_dir,'video_%d' % self.save_count,'image')
        else:
            save_dir = os.path.join(self.output_dir,self.dir_name+'_%d' % self.save_count,'image')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(os.path.join(save_dir,'depth')):
            os.mkdir(os.path.join(save_dir,'depth'))

        imageio.imwrite(os.path.join(save_dir,'depth','%d.png'% self.image_num), depth)
    
        return

    # Save video for the whole rendered data
    def save_video(self):
        if len(self.images) != 0:
            if self.dir_name == '':
                video_dir = os.path.join(self.output_dir,'video'+'_%d' % self.save_count)
                if not os.path.exists(video_dir):
                    os.mkdir(video_dir)
            else:
                video_dir = os.path.join(self.output_dir,self.dir_name+'_%d' % self.save_count)
                if not os.path.exists(video_dir):
                    os.mkdir(video_dir)

            imageio.mimwrite(video_dir + '/%s_color_%d.mp4' % (self.dir_name,self.save_count), self.images, fps = self.fps, quality = 8)
            if len(self.depths) != 0:
                imageio.mimwrite(video_dir + '/%s_depth_%d.mp4' % (self.dir_name,self.save_count), self.depths, fps = self.fps, quality = 8)
            else:
                print('Warning: depth buffer is empty.')
            self.save_count += 1
        else:
            print('Warning: Cannot generate video for all rendered images, data is empty.')

    # Helper functions
    def set_save_dir(self, dir_name):
        self.dir_name = dir_name

    def set_fps(self, fps):
        self.fps = fps

    def set_threshold_low(self, low):
        self.model.threshold_low = low

    def set_threshold_high(self, high):
        self.model.threshold_high = high
    
    def set_rendering_path(self, path):
        self.rendering_path = path

    def set_resolution(self, resolution):

        aspect = self.width / self.height

        # Resolution can not be lower than 256 pixels
        resolution = max(256, resolution)

        if aspect >= 1.0:
            self.height = int(resolution)
            self.width = int(resolution * aspect)
        else:
            self.width = int(resolution)
            self.height = int(resolution / aspect)
        
        self.dataset.set_resolution(self.width, self.height)


    def color(self, result, render_type='Gaussian'):
        if self.model.spacenet_fine is not None:
            color = result['fine_color'].reshape(self.height,self.width,-1).cpu()
        else:
            color = result['coarse_color'].reshape(self.height,self.width,-1).cpu()
        
        ray_mask = result['ray_mask'].reshape(self.height,self.width,-1).cpu()

        return color

    def depth(self, result):
        if self.model.spacenet_fine is not None:
            depth = result['fine_depth'].reshape(self.height,self.width,-1).cpu()
        else:
            depth = result['coarse_depth'].reshape(self.height,self.width,-1).cpu()

        return depth


    # Visualize the distribution along a ray
    def ray_distribution(self, view_id, u_min, u_max, v_min, v_max):
        print('Selecting (%d, %d) to (%d, %d) from resolution (%d, %d)' % (v_min, u_min, v_max, u_max, self.height, self.width))
        pose = self.dataset.poses[view_id]
        with torch.no_grad():
            rays, bbox, near_far = self.dataset.get_rays(pose=pose, 
                                                         area=[u_min / self.width , u_max / self.width , v_min / self.height , v_max / self.height], 
                                                         super_res=False)
            rays = rays.cuda(self.gpu)
            bbox = bbox.cuda(self.gpu)
            near_far = near_far.cuda(self.gpu)

            print(rays.shape, bbox.shape, near_far.shape)
            results = self.model(rays, bbox, near_far, save_pts=True)

            return results
            # results = batchify_ray(self.model, rays, bbox, near_far=near_far)

    def gradient(self, pts):
        pass
