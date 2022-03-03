import argparse
import os
import sys

sys.path.append('..')
from render import NeuralRenderer
import torch
from config import cfg

text = 'This is the program to render FVV, get help by -h.'
parser = argparse.ArgumentParser(description=text)
parser.add_argument('-c', '--config', default='', help='set the config file path to render the network')
parser.add_argument('-g','--gpu', type=int, default=0, help='set gpu id to render the network')
parser.add_argument('-t','--type', type=str, default='linear', help='set renderer type, linear or gt')
# Rendering setting
parser.add_argument('-fn', '--frame_number', type=int, default= 41, help='total frame of the video')
parser.add_argument('-fps','--fps',type=int, default=30, help='fps of FVV')
parser.add_argument('-dr','--downsample_rate', type=int, default=1, help='downsample rate, e.g, 8 means downsample 4096 to 512.')
parser.add_argument('-o','--output_name',type=str, default='video',help='set output video name')
args = parser.parse_args()

torch.cuda.set_device(args.gpu)
torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)

cfg.merge_from_file(args.config)
cfg.MODEL.USE_DEFORM_VIEW = False
cfg.DATASETS.FACTOR = args.downsample_rate
cfg.rendering = True

cfg.freeze()

renderer = NeuralRenderer(cfg)
renderer.set_save_dir(args.output_name)

if args.type == 'gt':
    renderer.set_fps(1)
    renderer.render_gt()

if args.type == 'linear':
    renderer.render_linear()

renderer.save_video()