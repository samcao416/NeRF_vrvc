#? Maybe useless?

import sys, os, math
# switch to "osmesa" or "egl" before loading pyrender
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from PIL import Image
import json

import matplotlib
import imageio

from tqdm import tqdm

def caculate_align_mat(d):
    view_dir = d # [batch_shape, 3]
    view_dir /= np.linalg.norm(view_dir, ord=2, axis=-1, keepdims=True) # normalize
    up_dir = np.broadcast_to(np.array([0., 1., 0.]), view_dir.shape) # [batch_shape, 3]

    # Grama-schmidta algorithm
    left_dir = np.cross(up_dir, view_dir, axis=-1) # [batch_shape, 3]
    left_dir /= np.linalg.norm(left_dir, ord=2, axis=-1, keepdims=True) # normalize
    up_dir = np.cross(view_dir, left_dir, axis=-1) # [batch_shape, 3]
    return np.stack([left_dir, up_dir, view_dir], -1) # [batch_shape, 3, 3]

def create_rays(o, d, c1, c2, size=0.75):
    positions = np.concatenate([o, o+d*size], -1).reshape(-1, 3)
    colors = np.concatenate([c1, c2], -1).reshape(-1, 3)
    return pyrender.Mesh([pyrender.Primitive(positions, color_0=colors, mode=pyrender.constants.GLTF.LINES)]) # Lines

def create_arrows(o, d, size=0.1, asize=0.01):
    positions = np.concatenate([o, o+d*size], -1).reshape(-1, 3)
    rot_mat = caculate_align_mat(d)
    d1 = rot_mat @ np.array([1, 0, -1.7]).reshape([3, 1]).squeeze(-1)
    d2 = rot_mat @ np.array([-1, 0, -1.7]).reshape([3, 1]).squeeze(-1)
    arrows1 = np.concatenate([o+d*size, o+d*size + asize*d1], -1).reshape(-1, 3)
    arrows2 = np.concatenate([o+d*size, o+d*size + asize*d2], -1).reshape(-1, 3)
    positions = np.concatenate([positions, arrows1, arrows2], 0)
    colors = np.zeros_like(positions)
    return pyrender.Mesh([pyrender.Primitive(positions, color_0=colors, mode=pyrender.constants.GLTF.LINES)]) # Lines

def create_path(o):

    x = np.linspace(0, 1, o.shape[0])
    cm = matplotlib.cm.get_cmap('Spectral_r')
    pts = pyrender.Primitive(o, color_0=cm(x), mode=pyrender.constants.GLTF.POINTS)

    colors = np.zeros_like(o)
    lines = pyrender.Primitive(o, color_0=colors, mode=pyrender.constants.GLTF.LINE_STRIP)
    return pyrender.Mesh([pts, lines])

class CameraLogger:

    def __init__(self, image_size=(800, 800), znear=0.1, zfar=None, yfov=np.pi/3,
        camera_pose=None, record_video=False, camera_path=False):
        
        W, H = image_size
        self.renderer = pyrender.OffscreenRenderer(W, H, point_size=20.)
        self.record_video = record_video
        self.camera_path = camera_path
        
        # set marker
        axis_marker = trimesh.creation.axis()
        self.axis_marker = pyrender.Mesh.from_trimesh(axis_marker, smooth=False)

        # set camera
        self.camera_pose = camera_pose
        if self.camera_pose is None:
            self.camera_pose = np.array([
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 2.5],
                [0., 0., 0., 1.]
            ])
        self.camera = pyrender.PerspectiveCamera(znear=znear, zfar=zfar, yfov=yfov)

        self.frames = []

    def log_img(self, rays_o, rays_d, ray_len=0.3):
        """
        Param:
            rays_o: [N_camera, 3]
            rays_d: [N_camera, 3]
            ray_len: the length of each rendered camera ray
        Return:
            Rendered camera poses [H, W, 3]
        """
        # create scene
        scene = pyrender.Scene(ambient_light=[.75, .75, .75], bg_color=[255, 255, 255])

        # add rays
        rays_primitive = create_arrows(rays_o, rays_d, ray_len)
        scene.add(rays_primitive, pose=np.eye(4))
        if self.camera_path:
            path_primitive = create_path(rays_o)
            scene.add(path_primitive, pose=np.eye(4))

        # add axis arrow
        scene.add(self.axis_marker)
        # add camera
        scene.add(self.camera, pose=self.camera_pose)

        color, depth = self.renderer.render(scene)
        if self.record_video:
            self.frames.append(color)

        return color

    def log_video(self, file_path, fps=1, quality=8):
        rgbs = np.stack(self.frames, 0)
        imageio.mimwrite(file_path, rgbs, fps=fps, quality=quality)

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('%s <cam_dir> [subsample]' % sys.argv[0])

    cam_dir = sys.argv[1]
    if not os.path.exists(cam_dir):
        print("[Error] Cameras not exist", cam_dir)
        exit(-1)

    subsample = 8
    if len(sys.argv) >= 3:
        subsample = sys.argv[2]


    # specify near and far below
    logger = CameraLogger(record_video=True)

    for filename in tqdm(sorted(list(os.listdir(cam_dir)))):
        file_path = os.path.join(cam_dir, filename)
        d = np.load(file_path)
        try:
            rays_o, rays_d = d['o'], d['d']
        except KeyError:
            rays_o, rays_d = d['rays_o'], d['rays_d']

        indices = np.linspace(0, rays_o.shape[0]-1, subsample, dtype=np.int64)
        rays_o, rays_d = rays_o[indices], rays_d[indices]

        logger.log_img(rays_o, rays_d)

    logger.log_video('camera_video.mp4', fps=1, quality=8)
