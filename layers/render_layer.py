import torch
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
        self.boarder_weight = boarder_weight #? so what's boarder_weight

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
        #print('in render_layer.py, color shape is: ', color.shape)
        #print('acc_map shape is: ', acc_map.shape)

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
        acc_map = torch.sum(alphas, dim = 1) #
        
        return color, depth, acc_map, alphas