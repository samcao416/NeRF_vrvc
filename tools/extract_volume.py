import argparse
import os
import sys

sys.path.append('..')

from data.datasets.volume import Volume
from modeling import build_model
from data import get_iteration_path_and_iter

import torch
from config import cfg

text = 'This is the program to extract 3D volume for Cryo-ET, get help by -h.'
parser = argparse.ArgumentParser(description=text)
parser.add_argument('-c', '--config', default='', help='set the config file path to render the network')
parser.add_argument('-g','--gpu', type=int, default=0, help='set gpu id to render the network')
# Rendering setting
parser.add_argument('-s', '--size', type=int, default= 1024, help='Slicing height (pixel)')
parser.add_argument('-a','--aspect',type=float, default=1.0, help='H over W')
parser.add_argument('-t','--thick', type=int, default=400, help='The number of slicing')
parser.add_argument('-o','--output_name',type=str, default='volume',help='set output volume name')
args = parser.parse_args()

# Load config
training_config = args.config
assert os.path.exists(training_config), 'training config does not exist.'
cfg.merge_from_file(training_config)

cfg.add_noise = False
cfg.noise_scale = 0
cfg.slice = False
cfg.rendering = False

torch.cuda.set_device(args.gpu)
torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)

W = args.size
H = int(args.size * args.aspect)
D = args.thick

# Create model, optimizer and scheduler
model = build_model(cfg, 0).cuda()
# Load from checkpoint with fixed iteration number or automatically find the maximum one.
para_file, iter0 = get_iteration_path_and_iter(cfg.OUTPUT_DIR)

    
assert os.path.exists(para_file) and para_file != "", 'The checkpoint does not exist:' + para_file
print('Loaded model from checkpoint: ',para_file)
dict_0 = torch.load(os.path.join(cfg.OUTPUT_DIR,para_file),map_location='cuda:%d' % int(args.gpu))
# Add non-existing parameters into pretrained model parameters
model_dict = dict_0['model']

model.load_state_dict(model_dict)

# Create an empty volume
volume = Volume(cfg, W, H, D, device=torch.device('cuda'))
# Update density from model
volume.update_density(model)

volume.save(os.path.join(cfg.OUTPUT_DIR, args.output_name+'.mrc'))
