import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def gen_weight(sigma, delta, act_fn=F.relu):
    """Generate transmittance from predicted density
    """
    alpha = 1.-torch.exp(-act_fn(sigma.squeeze(-1))*delta)
    weight = 1.-alpha + 1e-10
    #weight = alpha * torch.cumprod(weight, dim=-1) / weight # exclusive cum_prod

    weight = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1),device = alpha.device), weight], -1), -1)[:, :-1]

    return weight

def gen_alpha(sigma, delta, act_fn=F.relu):
    """Generate descritized alpha from predicted density and known delta
    """
    # alpha = 1.-torch.exp(-act_fn(sigma.squeeze(-1))*delta)
    # TODO: below line seems to be no big effects
    alpha = act_fn(sigma.squeeze(-1))*delta
    # alpha = sigma.squeeze(-1) * delta

    return alpha

class VolumeRenderer(nn.Module):
    def __init__(self, boarder_weight = 1e10):
        super(VolumeRenderer, self).__init__()
        self.boarder_weight = boarder_weight 

    def forward(self, depth, rgb, sigma, noise=0):
        """
        N - num rays; L - num samples; 
        :param depth: torch.tensor, depth for each sample along the ray. [N, L, 1]
        :param rgb: torch.tensor, raw rgb output from the network. [N, L, 3]
        :param sigma: torch.tensor, raw density (without activation). [N, L, 1]
        
        :return:
            color: torch.tensor [N, 3] 
            depth: torch.tensor [N, 1]
        """

        delta = (depth[:, 1:] - depth[:, :-1]).squeeze()  # [N, L-1]
        #pad = torch.Tensor([1e10],device=delta.device).expand_as(delta[...,:1])
        pad = self.boarder_weight*torch.ones(delta[...,:1].size(),device = delta.device)
        delta = torch.cat([delta, pad], dim=-1)   # [N, L]

        if noise > 0.:
            sigma += (torch.randn(size=sigma.size(),device = delta.device) * noise)

        weights = gen_weight(sigma, delta).unsqueeze(-1)    #[N, L, 1]

        color = torch.sum(torch.sigmoid(rgb) * weights, dim=1) #[N, 3]
        depth = torch.sum(weights * depth, dim=1)   # [N, 1]
        acc_map = torch.sum(weights, dim = 1) # [N, 1]

        return color, depth, acc_map, weights
    
class Projector(nn.Module):

    def __init__(self):
        super(Projector, self).__init__()

    def forward(self, depth, color, density):
        """
        N - num rays; L - num samples; 
        :param depth: torch.tensor, depth for each sample along the ray. [N, L, 1]
        :param rgb: torch.tensor, raw rgb output from the network. [N, L, 3]
        :param sigma: torch.tensor, raw density (without activation). [N, L, 1]
        
        :return:
            color: torch.tensor [N, 3]
            depth: torch.tensor [N, 1]
        """

        delta = (depth[:, 1:] - depth[:, :-1]).squeeze()  # [N, L-1]
        #pad = torch.Tensor([1e10],device=delta.device).expand_as(delta[...,:1])
        pad = 0 * torch.ones(delta[...,:1].size(),device = delta.device)
        delta = torch.cat([delta, pad], dim=-1)   # [N, L]
        #TODO: Check
        alphas = gen_alpha(density, delta).unsqueeze(-1)

        # weights = weights.unsqueeze(-1) # (N, L, 1)

        color = torch.sum(alphas, dim=1) #[N, NUM_CHANNEL]
        # Expected depth
        depth = torch.sum(alphas * depth, dim=1) / (color + 1e-5)  # [N, 1]
        acc_map = color
        
        return color, depth, acc_map, alphas

class AlphaBlender(nn.Module):
    def __init__(self):
        super(AlphaBlender, self).__init__()

    def forward(self, depth, color, density):
        """
        N - num rays; L - num samples; 
        :param depth: torch.tensor, depth for each sample along the ray. [N, L, 1]
        :param rgb: torch.tensor, raw rgb output from the network. [N, L, 3]
        :param sigma: torch.tensor, raw density (without activation). [N, L, 1]
        
        :return:
            color: torch.tensor [N, 3]
            depth: torch.tensor [N, 1]
        """

        delta = (depth[:, 1:] - depth[:, :-1]).squeeze()  # [N, L-1]
        #pad = torch.Tensor([1e10],device=delta.device).expand_as(delta[...,:1])
        pad = 0*torch.ones(delta[...,:1].size(),device = delta.device)
        delta = torch.cat([delta, pad], dim=-1)   # [N, L]

        alphas = gen_alpha(density, delta).unsqueeze(-1)    #[N, L, 1]
        alphas = F.normalize(alphas, p=1, dim=1)

        color = torch.sum(torch.sigmoid(color) * alphas, dim=1) #[N, 3]
        depth = torch.sum(alphas * depth, dim=1)   # [N, 1]
        acc_map = torch.sum(alphas, dim = 1) # [N, 1]
        
        return color, depth, acc_map, alphas

class VolumeRendererMip(nn.Module):
    def __init__(self, 
                 white_bg = False,
                 randomized = True, 
                 density_noise: float = 1.,
                 density_bias:  float = -1.,
                 rgb_padding:   float = 0.0001):
        super(VolumeRendererMip, self).__init__()
        self.white_bg = white_bg
        self.density_noise = density_noise
        self.density_bias  = density_bias
        self.rgb_padding   = rgb_padding
        self.randomized    = randomized

    def forward(self, density, rgbs, t_vals, dirs):
        if self.randomized and self.density_noise > 0:
            density += self.density_noise * torch.randn(
                *(density.shape), dtype = density.dtype, device = density.device
            ) # N, L, 1
        rgbs = torch.sigmoid(rgbs) # N, L, 3

        density = F.softplus(density + self.density_bias) # N, L, 1

        t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:]) # N, L
        t_dists = t_vals[..., 1:] - t_vals[..., :-1] # N, L
        delta = t_dists * torch.linalg.norm(dirs[..., None, :], dim = -1) # N, L
        
        #Note that we're quietly turning density from [..., 0] to [...].
        density_delta = density[..., 0] * delta # N, L

        alpha = 1 - torch.exp(-density_delta) # N, L
        trans = torch.exp(-torch.cat([
            torch.zeros_like(density_delta[..., :1]),
            torch.cumsum(density_delta[..., :-1], dim = -1)
        ],
            dim = -1)) # N, L
        weights = alpha * trans # N, L

        comp_rgb = (weights[..., None] * rgbs).sum(dim = -2) # N, 3
        acc = weights.sum(dim = -1) # N,
        depth = (weights * t_mids).sum(dim = -1) / acc
        depth = torch.clip(
            torch.nan_to_num(depth, float('inf')), t_vals[:, 0], t_vals[:, -1]
        ) # N,
        if self.white_bg:
            comp_rgb = comp_rgb + (1. - acc[..., None])

        #comp_rgb = comp_rgb.unsqueeze(-1) # N, 3, 1
        depth = depth.unsqueeze(-1) # N, 1
        acc = acc.unsqueeze(-1) # N, 1
        weights = weights.unsqueeze(-1) # N, L, 1
        return comp_rgb, depth, acc, weights