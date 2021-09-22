import torch
import numpy as np
import mrc
from tqdm import tqdm, trange

class Volume(torch.utils.data.Dataset):
    
    def __init__(self, cfg, H, W, D, device=torch.device('cpu')):
        
        # Save input parameters
        self.cfg = cfg
        self.device = device
        self.H = H
        self.W = W
        self.D = D

        # Global bounding box
        self.volume_size = (H, W, D)
        self.extents = (2, 2, self.cfg.DATASETS.SAMPLE_HEIGHT)

        # Create positions of 3D volume (H, W, D, 3) -> (H x W x D, 3)
        self.pos = self.create_volume(self.extents, self.volume_size, device=self.device).reshape(-1,3)
        print(self.pos)

        # Create empty density volume
        self.density = torch.zeros(H*W*D,1)

    def create_volume(self, extents, num_grids, anchors='center', device=torch.device('cpu')):
        # `joint` is chosen from:
        # 1. lower: the point is sharply sampled on lower/left corner pixel
        # 2. center: the joint point is sampled on the center of corner pixel
        # 3. upper: the point is sharply sampled on upper/right corner pixel
        assert anchors in ['lower', 'center', 'upper']

        if np.isscalar(num_grids):
            num_grids = [num_grids] * len(extents)
        assert len(extents) == len(num_grids)

        extents = np.array(extents, dtype=np.float32) # length of each axis
        num_grids = np.array(num_grids, dtype=np.int32) # number of grid points for axes

        # sample lattice at the lower corner by default
        starts = -extents / 2
        ends = extents / 2
        pixel_sizes = extents / num_grids

        if anchors == 'lower':
            ends -= pixel_sizes
        elif anchors == 'center':
            starts += pixel_sizes / 2.
            ends -= pixel_sizes / 2.
        elif anchors == 'upper':
            starts += pixel_sizes

        xyz = torch.meshgrid(*[
            torch.linspace(s, e, n, device=device) for (s, e, n) in zip(starts, ends, num_grids)
        ])
        lattice = torch.stack(xyz, -1) # [N_1, N_2, ..., N_m, m]

        return lattice

    def __length__(self):
        return self.pos.shape[0]

    def __getitem__(self, index):
        return self.pos[index], self.density[index]

    def save(self, save_dir, meta_data={}, apix=1.0, title=None):

        volume = self.density.reshape(self.H, self.W, self.D).permute(2,1,0).cpu().numpy()
        volume = np.flip(volume, 2)

        meta_new = {
            'm': np.array(volume.shape, dtype=np.float32),
            'd': np.array(volume.shape, dtype=np.float32) * apix,
            'wave': np.array([0., 0., 0., 0., 0.]),
        }
        # meta_new.update(meta_data)
        if title is not None:
            meta_new['NumTitles'] = 1
            meta_new['title'] = title

        mrc.imwrite(save_dir, volume, metadata=meta_new)

    def update_density(self, model):
        with torch.no_grad():
            for i in trange(0, self.pos.shape[0], self.H * self.W):
                end = min(self.pos.shape[0], i+self.H * self.W)
                pts = self.pos[i:end]
                self.density[i:end] = model.get_density(pts)
