import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.autograd.set_detect_anomaly(True)

class PositionalEncoder(nn.Module):
    def __init__(self, dim_in, num_freqs, max_freq=-1, log_sampling=True, include_input=True):
        super(PositionalEncoder, self).__init__()

        if max_freq < 0:
            max_freq = num_freqs - 1

        self.dim_in = dim_in
        self.num_freqs = num_freqs
        self.max_freq = max_freq
        self.include_input = include_input
        self.log_sampling = log_sampling

        if log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=num_freqs)
        else:
            freq_bands = torch.linspace(0., 2.**max_freq, steps=num_freqs)
        self.register_buffer('freq_bands', freq_bands, persistent=False)

        self.dim_out = num_freqs * self.dim_in * 2
        if self.include_input:
            self.dim_out += self.dim_in

    def embed_size(self):
        return self.dim_out

    def forward(self, x):
        """
        Args:
        x: [N_pts, 3] point coordinates
        Return:
        embedded results of out_size() dimension
        """
        freq_bands = self.freq_bands.expand(1, 1, self.num_freqs) # [1, 1, N_freqs]
        y = x[..., None] * freq_bands  # [N_pts, dim_in, N_freqs]
        y = y.view(x.shape[0], -1) # [N_pts, dim_in * N_freqs]
        if self.include_input:
            return torch.cat([x, torch.sin(y), torch.cos(y)], -1)
        else:
            return torch.cat([torch.sin(y), torch.cos(y)], -1)

class FourierFeatEncoder(nn.Module):

    def __init__(self, dim_in, dim_embed, ffm_scale=16., trainable=False):
        super(FourierFeatEncoder, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_embed * 2

        B = torch.normal(0., ffm_scale, size=(dim_embed, dim_in))
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)

    def embed_size(self):
        return self.dim_out

    def forward(self, x):
        y = torch.matmul(x, self.B.T)
        return torch.cat([torch.sin(y), torch.cos(y)], dim=-1)

class RandFourierFeatEncoder(nn.Module):
    '''
    Generate Random Fourier Features (RFF) corresponding to the kernel:
        k(x, y) = k_a(x_a, y_a)*k_b(x_b, y_b)
    where 
        k_a(x_a, y_a) = exp(-\norm(x_a-y_a)/gamma_1),
        k_b(x_b, y_b) = <x_b, y_b>^gamma_2.
    '''
    def __init__(self, dim_in, dim_embed, kernel, trainable=False, **kwargs):
        super(RandFourierFeatEncoder, self).__init__()

        from utils.sampling import exp_sample, exp2_sample, matern_sample, \
            gamma_exp2_sample, rq_sample, poly_sample

        if kernel == 'exp1':
            W = exp_sample(kwargs['length_scale'], dim_embed)
        elif kernel == 'exp2':
            W = exp2_sample(kwargs['length_scale'], dim_embed)
        elif kernel == 'matern':
            W = matern_sample(kwargs['length_scale'], kwargs['matern_order'], dim_embed)
        elif kernel == 'gamma_exp':
            W = gamma_exp2_sample(kwargs['length_scale'], kwargs['gamma_order'], dim_embed)
        elif kernel == 'rq':
            W = rq_sample(kwargs['length_scale'], kwargs['rq_order'], dim_embed)
        elif kernel == 'poly':
            W = poly_sample(kwargs['poly_order'], dim_embed)
        else:
            raise NotImplementedError('Unknow RFF kernel:', kernel)
        b = np.random.uniform(0, np.pi * 2, dim_embed)

        if trainable:
            self.W = nn.Parameter(torch.tensor(W))
            self.b = nn.Parameter(torch.tensor(b))
        else:
            self.register_buffer('W', torch.tensor(W))
            self.register_buffer('b', torch.tensor(b))

    def embed_size(self):
        return self.W.shape[0]

    def forward(self, X):
        y = torch.matmul(x, self.W.T) + self.b
        return torch.cos(y)
