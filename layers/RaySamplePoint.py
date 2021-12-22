import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
from layers.render_layer import gen_weight
import pdb

#TODO: to add a description of this function
def intersection(rays, bbox):
    n = rays.shape[0]
    left_face = bbox[:, 0, 0]
    right_face = bbox[:, 6, 0]
    front_face = bbox[:, 0, 1]
    back_face = bbox[:, 6, 1]
    bottom_face = bbox[:, 0, 2]
    up_face = bbox[:, 6, 2]
    
    left_t = ((left_face - rays[:, 0]) / (rays[:, 3] + np.finfo(float).eps.item())).reshape((n, 1))
    right_t = ((right_face - rays[:, 0]) / (rays[:, 3] + np.finfo(float).eps.item())).reshape((n, 1))
    front_t = ((front_face - rays[:, 1]) / (rays[:, 4] + np.finfo(float).eps.item())).reshape((n, 1))
    back_t = ((back_face - rays[:, 1]) / (rays[:, 4] + np.finfo(float).eps.item())).reshape((n, 1))
    bottom_t = ((bottom_face - rays[:, 2]) / (rays[:, 5] + np.finfo(float).eps.item())).reshape((n, 1))
    up_t = ((up_face - rays[:, 2]) / (rays[:, 5] + np.finfo(float).eps)).reshape((n, 1))


    rays_o = rays[:, :3]
    rays_d = rays[:, 3:6]
    
    left_point = left_t * rays_d + rays_o
    right_point = right_t * rays_d + rays_o
    front_point = front_t * rays_d + rays_o
    back_point = back_t * rays_d + rays_o
    bottom_point = bottom_t * rays_d + rays_o
    up_point = up_t * rays_d + rays_o

    left_mask = (left_point[:, 1] >= bbox[:, 0, 1]) & (left_point[:, 1] <= bbox[:, 7, 1]) \
                & (left_point[:, 2] >= bbox[:, 0, 2]) & (left_point[:, 2] <= bbox[:, 7, 2])
    right_mask = (right_point[:, 1] >= bbox[:, 1, 1]) & (right_point[:, 1] <= bbox[:, 6, 1]) \
                 & (right_point[:, 2] >= bbox[:, 1, 2]) & (right_point[:, 2] <= bbox[:, 6, 2])

    # compare x, z
    front_mask = (front_point[:, 0] >= bbox[:, 0, 0]) & (front_point[:, 0] <= bbox[:, 5, 0]) \
                 & (front_point[:, 2] >= bbox[:, 0, 2]) & (front_point[:, 2] <= bbox[:, 5, 2])

    back_mask = (back_point[:, 0] >= bbox[:, 3, 0]) & (back_point[:, 0] <= bbox[:, 6, 0]) \
                & (back_point[:, 2] >= bbox[:, 3, 2]) & (back_point[:, 2] <= bbox[:, 6, 2])

    # compare x,y
    bottom_mask = (bottom_point[:, 0] >= bbox[:, 0, 0]) & (bottom_point[:, 0] <= bbox[:, 2, 0]) \
                  & (bottom_point[:, 1] >= bbox[:, 0, 1]) & (bottom_point[:, 1] <= bbox[:, 2, 1])

    up_mask = (up_point[:, 0] >= bbox[:, 4, 0]) & (up_point[:, 0] <= bbox[:, 6, 0]) \
              & (up_point[:, 1] >= bbox[:, 4, 1]) & (up_point[:, 1] <= bbox[:, 6, 1])

    tlist = -torch.ones_like(rays, device=rays.device)*1e3
    tlist[left_mask, 0] = left_t[left_mask].reshape((-1,))
    tlist[right_mask, 1] = right_t[right_mask].reshape((-1,))
    tlist[front_mask, 2] = front_t[front_mask].reshape((-1,))
    tlist[back_mask, 3] = back_t[back_mask].reshape((-1,))
    tlist[bottom_mask, 4] = bottom_t[bottom_mask].reshape((-1,))
    tlist[up_mask, 5] = up_t[up_mask].reshape((-1,))
    tlist = tlist.topk(k=2, dim=-1)

    return tlist[0]



def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable = True):
    if stable:
      mu = (t0 + t1) / 2
      hw = (t1 - t0) / 2
      t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
      t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) /
                                        (3 * mu**2 + hw**2)**2)
      r_var = base_radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 *
                                (hw**4) / (3 * mu**2 + hw**2))
    else:
      t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
      r_var = base_radius**2 * (3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3))
      t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
      t_var = t_mosq - t_mean**2
    return lift_gaussian(d, t_mean, t_var, r_var, diag)

def cylinder_to_gaussian(d, t0, t1, radius, diag):
    t_mean = (t0 + t1) / 2
    r_var = radius**2 / 4
    t_var = (t1 - t0)**2 / 12
    return lift_gaussian(d, t_mean, t_var, r_var, diag)
    
def lift_gaussian(d, t_mean, t_var, r_var, diag):
    mean = d[..., None, :] * t_mean[..., None]
    small_tensor = torch.ones_like(torch.sum(d**2, dim=-1, keepdims=True)) * 1e-10
    d_mag_sq = torch.maximum(small_tensor, torch.sum(d**2, dim=-1, keepdims=True))
    if diag:
        d_outer_diag = d**2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]
        eye = torch.eye(d.shape[-1])
        null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
        cov = t_cov + xy_cov
        return mean, cov

def cast_rays(t_vals, rays, radii, ray_shape, diag = True):
    origins = rays[:,0:3]
    t0 = t_vals[..., :-1]
    t1 = t_vals[..., 1:]
    if ray_shape == 'cone':
        gaussian_fn = conical_frustum_to_gaussian
    elif ray_shape == 'cylinder':
        gaussian_fn = cylinder_to_gaussian
    else:
        assert False
    means, covs = gaussian_fn(rays[:,3:6], t0, t1, radii, diag)
    means = means + origins[..., None, :]
    return means, covs
    
def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized):
    """Piecewise-Constant PDF sampling from sorted bins.
    Args:
        key: jnp.ndarray(float32), [2,], random number generator.
        bins: jnp.ndarray(float32), [batch_size, num_bins + 1].
        weights: jnp.ndarray(float32), [batch_size, num_bins].
        num_samples: int, the number of samples.
        randomized: bool, use randomized samples.
    Returns:
        t_samples: jnp.ndarray(float32), [batch_size, num_samples].
    """
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    eps = 1e-5
    weight_sum = torch.sum(weights, dim=-1, keepdim=True)
    padding = torch.maximum(torch.zeros_like(eps - weight_sum), eps - weight_sum)
    weights = weights + padding / weights.shape[-1]
    weight_sum = weight_sum + padding   
    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum
    cdf = torch.minimum(torch.ones_like(torch.cumsum(pdf[..., :-1], dim=-1)), torch.cumsum(pdf[..., :-1], dim=-1))
    cdf = torch.cat([
        torch.zeros(list(cdf.shape[:-1]) + [1], device=cdf.device), cdf,
        torch.ones(list(cdf.shape[:-1]) + [1], device=cdf.device)
    ],
                          dim=-1)

    # Draw uniform samples.
    if randomized:
        s = 1 / num_samples
        u = torch.arange(num_samples, device=cdf.device) * s
        u = u + torch.empty(size=list(cdf.shape[:-1]) + [num_samples], device=cdf.device).uniform_(to=s - torch.finfo(torch.float32).eps)
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.ones_like(u) - torch.finfo(torch.float32).eps)
    else:
        # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
        u = torch.linspace(0., 1. - torch.finfo(torch.float32).eps, num_samples)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])   
    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.
    mask = u[..., None, :] >= cdf[..., :, None] 
    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0, _ = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
        x1, _ = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)
        return x0, x1 
    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf) 
    t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    return samples
                        
class RaySamplePoint(nn.Module):
    def __init__(self, coarse_num=64, noise_level = 0):
        super(RaySamplePoint, self).__init__()
        self.coarse_num = coarse_num
        self.noise_level = noise_level


    def forward(self, rays, bbox, pdf=None,  method='coarse', noise_level = 0):
        '''
        :param rays: N*6
        :param bbox: N*8*3  0,1,2,3 bottom 4,5,6,7 up        
        pdf: n*coarse_num 表示权重
        :param method:
        :return: N*C*1  ,  N*C*3,   N   C for counts or coarse
        '''
        n = rays.shape[0]
        #if method=='coarse':
        sample_num = self.coarse_num
        bin_range = torch.arange(0, sample_num, device=rays.device).reshape((1, sample_num)).float() #bin?

        bin_num = sample_num
        n = rays.shape[0]
        tlist = intersection(rays, bbox)
        start = (tlist[:,1]).reshape((n,1))   
        end = (tlist[:, 0]).reshape((n, 1))
        
        bin_sample = torch.rand((n, sample_num), device=rays.device)
        bin_width = (end - start)/bin_num
        sample_t = (bin_range + bin_sample)* bin_width + start
        
        # there is a y-axis noise to make a connection between different pixel rows. 
        
        # For every ray
        # y_noise = torch.ones_like(rays[:,0]) * self.noise_level * np.random.normal(0,1,size=(1,1)).item()
        
        # TODO: For every single point
        #y_noise = torch.tensor(self.noise_level * np.random.normal(0,1,size=(rays.shape[0],)), device=rays.device)
        
        pos = rays[:,:3]
        pos[:,1] = pos[:,1]
        
        sample_point = sample_t.unsqueeze(-1)*rays[:,3:6].unsqueeze(1) + pos.unsqueeze(1)
        mask = (torch.abs(bin_width)> 1e-5).squeeze()
        return sample_t.unsqueeze(-1), sample_point, mask

    def set_coarse_sample_point(self, a):
        self.coarse_num = a


class RayDistributedSamplePoint(nn.Module):
    def __init__(self, fine_num=10):
        super(RayDistributedSamplePoint, self).__init__()
        self.fine_num = fine_num

    def forward(self, rays, depth, density, noise=0.0):
        '''
        :param rays: N*L*6
        :param depth: N*L*1
        :param density: N*L*1
        :param noise:0
        :return:
        ''' 

        sample_num = self.fine_num
        n = density.shape[0]

        weights = gen_weight(depth, density, noise=noise) # N*L
        weights += 1e-5
        bin = depth.squeeze()

        weights = weights[:, 1:].squeeze() #N*(L-1)
        pdf = weights/torch.sum(weights, dim=1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=1)
        cdf_s = torch.cat((torch.zeros((n, 1)).type(cdf.dtype), cdf), dim=1)
        fine_bin = torch.linspace(0, 1, sample_num, device=density.device).reshape((1, sample_num)).repeat((n, 1))
        above_index = torch.ones_like(fine_bin, device=density.device).type(torch.LongTensor)
        for i in range(cdf.shape[1]):
            mask = (fine_bin > (cdf_s[:, i]).reshape((n, 1))) & (fine_bin <= (cdf[:, i]).reshape((n, 1)))
            above_index[mask] = i+1
        below_index = above_index-1
        below_index[below_index==-1]=0
        sn_below = torch.gather(bin, dim=1, index=below_index)
        sn_above = torch.gather(bin, dim=1, index=above_index)
        cdf_below = torch.gather(cdf_s, dim=1, index=below_index)
        cdf_above = torch.gather(cdf_s, dim=1, index=above_index)
        dnorm = cdf_above - cdf_below
        dnorm = torch.where(dnorm<1e-5, torch.ones_like(dnorm, device=density.device), dnorm)
        d = (fine_bin - cdf_below)/dnorm
        fine_t = (sn_above - sn_below) * d + sn_below
        fine_sample_point = fine_t.unsqueeze(-1) * rays[:, 3:6].unsqueeze(1) + rays[:, :3].unsqueeze(1)
        return fine_t, fine_sample_point



class RaySamplePoint_Near_Far(nn.Module):
    def __init__(self, r = 2 * 5 / 1024, sigma = 2 * 5 / 1024, sample_num=75, noise_level = 0):
        super(RaySamplePoint_Near_Far, self).__init__()
        self.sample_num = sample_num
        self.r = r
        self.sigma = sigma
        self.noise_level = noise_level


    def forward(self, rays,near_far):
        '''
        :param rays: N*6
        :param bbox: N*8*3  0,1,2,3 bottom 4,5,6,7 up
        pdf: n*coarse_num 表示权重
        :param method:
        :return: N*C*3
        '''
        n = rays.size(0)
        

        ray_o = rays[:,:3]
        ray_d = rays[:,3:6]

        t_vals = torch.linspace(0., 1., steps=self.sample_num,device =rays.device)
        z_vals = near_far[:,0:1].repeat(1, self.sample_num) * (1.-t_vals).unsqueeze(0).repeat(n,1) +\
                 near_far[:,1:2].repeat(1, self.sample_num) * (t_vals.unsqueeze(0).repeat(n,1))

        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # ZJK Version:
        z_vals = (lower + upper) / 2
        # WMY Version:
        # t_rand = torch.rand(z_vals.size(), device = rays.device)
        # z_vals = lower + (upper - lower) * t_rand #batch_size * sample_num
        
        pos = ray_o

        pts = pos[...,None,:] + ray_d[...,None,:] * z_vals[...,:,None]
        
        return z_vals.unsqueeze(-1), pts #[batch_size, sample_num, 1], [batch_size, sample_num, 3]

    def set_coarse_sample_point(self, a):
        self.sample_num = a

class RaySamplePoint_Mip(nn.Module):
    def __init__(self, 
                 sample_num = 75, 
                 ray_shape = "cylinder", 
                 lindisp = False, 
                 use_viewdirs = True,
                 randomized = True):
        super(RaySamplePoint_Mip, self).__init__()
        self.sample_num = sample_num
        self.ray_shape = ray_shape
        self.lindisp = lindisp
        self.use_viewdirs = use_viewdirs
        self.randomized = randomized


    def forward(self, rays, radii, near_far):
        batch_size = rays.shape[0]
        device = rays.device

        t_vals = torch.linspace(0., 1., setps = self.sample_num + 1, device = device)
        if self.lindisp:
            t_vals = 1. / (1. / near_far[:,0:1] * (1. - t_vals) + 1. / near_far[:,1:2] * t_vals) # N, sample_num + 1
        else:
            t_vals = near_far[:,0:1] * (1. - t_vals) + near_far[:, 1:2] * t_vals # N, sample_num + 1
        
        if self.randomized:
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], -1)
            lower = torch.cat([t_vals[..., 1], mids], -1)
            t_rand = torch.rand(batch_size, self.sample_num + 1, device = device)
            t_vals = lower + (upper - lower) * t_rand
        else:
            t_vals = torch.broadcast_to(t_vals, [batch_size, self.sample_num + 1])
        means, covs = cast_rays(t_vals, rays, radii, self.ray_shape)
        return t_vals, (means, covs) # [batch_size, sample_num + 1], ([batch_size, sample_num, 3], [batch_size, sample_num, 3])

    def set_coarse_sample_point(self, a):
        self.sample_num = a
        
class ResamplePointMip(nn.Module):
    def __init__(self,
                 sample_num = 75,
                 randomized = True,
                 ray_shape  = 'cylinder',
                 stop_level_grad = True,
                 resample_padding = 0.01
                 ):
        super(ResamplePointMip, self).__init__()
        self.sample_num = sample_num
        self.randomized = randomized
        self.ray_shape  = ray_shape
        self.stop_level_grad = stop_level_grad
        self.resample_padding = resample_padding

    def forward(self, rays, radii, weights, t_vals):
        weights = weights.squeeze(-1)
        weights_pad = torch.cat([
            weights[..., :1],
            weights,
            weights[..., -1:]
        ], dim = -1) # N, L + 2

        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:]) # N, L +1
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:]) # N, L

        weights = weights_blur + self.resample_padding # N, L

        new_t_vals = sorted_piecewise_constant_pdf(t_vals,  # N, L + 1
                                                   weights, # N, L
                                                   t_vals.shape[-1], # = L + 1
                                                   self.randomized)
        
        if self.stop_level_grad:
            new_t_vals = new_t_vals.detach()
        means, covs = cast_rays(new_t_vals, rays, radii, self.ray_shape)
        return new_t_vals, (means, covs)