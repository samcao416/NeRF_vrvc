import torch
import numpy as np

def meshgrid_xy(tensor1: torch.Tensor, tensor2: torch.Tensor):
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1,- 2), jj.transpose(-1, -2)

def ray_sampling_syn(image, pose, focal):
    H, W = image.shape[0:2]
    ii, jj = meshgrid_xy(
        torch.arange(W),torch.arange(H)
        )
    dirs = torch.stack(
        [
            (ii - W * 0.5) / focal,
            -(jj - H * 0.5) / focal,
            torch.ones_like(ii)
        ],
        dim = -1
    )
    ray_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim = -1)
    ray_o = pose[:3, -1].expand(ray_d.shape)
    rays = torch.cat([ray_o, ray_d], dim = -1)
    rays = rays.reshape(-1,6)

    colors = torch.Tensor(image.reshape(-1,3))
    return rays, colors

def generate_rays_syn(H, W, pose, focal):
    ii, jj = meshgrid_xy(
        torch.arange(W),torch.arange(H)
        )
    dirs = torch.stack(
        [
            (ii - W * 0.5) / focal,
            -(jj - H * 0.5) / focal,
            torch.ones_like(ii)
        ],
        dim = -1
    )
    ray_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim = -1)
    ray_o = pose[:3, -1].expand(ray_d.shape)
    rays = torch.cat([ray_o, ray_d], dim = -1)
    rays = rays.reshape(-1,6)

    return rays

def ray_sampling_mip(image, pose, focal):
    H, W = image.shape[0:2]
    ii, jj = meshgrid_xy(
        torch.arange(W),torch.arange(H)
        )
    dirs = torch.stack(
        [
            (ii - W * 0.5) / focal,
            -(jj - H * 0.5) / focal,
            torch.ones_like(ii)
        ],
        dim = -1
    )
    ray_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim = -1)
    ray_o = pose[:3, -1].expand(ray_d.shape)
    rays = torch.cat([ray_o, ray_d], dim = -1)
    rays = rays.reshape(-1,6)

    dx = np.sqrt(torch.sum((ray_d[:-1,:,:] - ray_d[1:,:,:]) ** 2, -1))
    dx = np.concatenate([dx, dx[-2:-1, :]], 0)
    radii = torch.Tensor(dx[..., None] * 2 / np.sqrt(12))
    lossmult = torch.ones_like(ray_o[..., :1])

    colors = torch.Tensor(image.reshape(-1,3))
    return rays, colors, radii.reshape(-1,1), lossmult.reshape(-1,1)

def generate_rays_mip(H, W, pose, focal):
    ii, jj = meshgrid_xy(torch.arange(W), torch.arange(H))
    dirs = torch.stack([
        (ii - W * 0.5) / focal,
        -(jj - H * 0.5) / focal,
        torch.ones_like(ii)
        ],
        dim = -1
    )
    ray_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim = -1)
    ray_o = pose[:3, -1].expand(ray_d.shape)
    rays = torch.cat([ray_o, ray_d], dim = -1)

    return rays
    

