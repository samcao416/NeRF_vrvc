import torch
import math
import random

from torch._C import device

def safe_trig_helper(x, fn, t = 100 * math.pi):
    return fn(torch.where(torch.abs(x) < t, x, x % t))

def safe_sin(x):
    return safe_trig_helper(x, torch.sin)

def safe_cos(x):
    return safe_trig_helper(x, torch.cos)

def expected_sin(x, x_var):
    y = torch.exp(-0.5 * x_var) * safe_sin(x)
    y_var = torch.maximum(
        torch.zeros_like(x_var, device = x_var.device), 0.5 * (1 - torch.exp(-2 * x_var) * safe_cos(2 * x)) - y ** 2
    )
    return y, y_var

def integrated_pos_enc(x_coord, min_deg, max_deg, diag = True):
    if diag:
        x, x_cov_diag = x_coord
        scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)], device = x.device)
        x_shape = list(x.shape[:-1]) + [-1]
        y = torch.reshape(x[..., None, :] * scales[:, None], x_shape)
        y_var = torch.reshape(x_cov_diag[..., None, :] * scales[:, None] ** 2, x_shape)
    else:
        x, x_cov = x_coord
        num_dims = x.shape[-1]
        basis = torch.cat(
            [2 ** i * torch.eye(num_dims) for i in range(min_deg, max_deg)], -1
        )
        y = torch.matmul(x, basis)
        y_var = torch.sum((torch.matmul(x_cov, basis)) * basis, -2)
    
    return expected_sin(
        torch.cat([y, y + 0.5 * math.pi], dim = -1),
        torch.cat([y_var] * 2, dim = -1))[0]

def pos_enc(x, min_deg, max_deg, append_identity = True):
    scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)], device = x.device)
    xb = torch.reshape((x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim = -1))
    if append_identity:
        return torch.cat([x] + [four_feat], dim = -1)
    else:
        return four_feat