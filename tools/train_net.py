import os
import sys
import shutil

sys.path.append('..')
from config import cfg
from data import make_ray_data_loader, make_ray_data_loader_view, get_iteration_path_and_iter
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer, WarmupMultiStepLR, build_scheduler
from layers import make_loss

from utils.logger import setup_logger
from layers.RaySamplePoint import RaySamplePoint

from torch.utils.tensorboard import SummaryWriter
import torch

import argparse

# Parse arguments
text = 'This is the program to train the nerf, try to get help by using -h'
parser = argparse.ArgumentParser(description=text)
parser.add_argument('-c', '--config', default='', help='set the config file path to train the network')
parser.add_argument('-g','--gpu', type=int, default=0, help='set gpu id to train the network')
parser.add_argument('-r','--resume', type=int, default = 0, help='set the checkpoint number to resume training')
parser.add_argument('-s','--psnr_thres', type=float, default = 100.0, help= 'set the psnr threshold to train next frame')
parser.add_argument('-cont','--cont',action="store_true", help='automatically continue to training')
parser.add_argument('-noise','--add_noise', type=float, default = 0.0, help= 'set noise level, default is zero')
parser.add_argument('-clean','--clean_ray', action='store_true', help='clear temporal rays')
parser.add_argument('-vis', '--visualization', type=bool, default = False, help = 'visualize all the rays')

args = parser.parse_args()

# Set PyTorch GPU id and settings
torch.cuda.set_device(args.gpu)
torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)

# Load config
training_config = args.config
assert os.path.exists(training_config), 'training config does not exist.'
cfg.merge_from_file(training_config)

# Dynamically added
cfg.clean_ray = args.clean_ray
cfg.rendering = False

cfg.freeze()

# Initialize writer and logger
output_dir = cfg.OUTPUT_DIR
writer = SummaryWriter(log_dir=output_dir, max_queue=1)
writer.add_text('OUT_PATH', output_dir,0)
logger = setup_logger("NERFRender", output_dir, 0)
logger.info("Running with config:\n{}".format(cfg))

# Save training config
shutil.copyfile(training_config, cfg.OUTPUT_DIR + '/config.yml')

# Create ray dataset
train_loader, dataset = make_ray_data_loader(cfg)
val_loader, dataset_val = make_ray_data_loader_view(cfg)
if args.visualization:
    dataset.vis(cfg)

# Create model, optimizer and scheduler
model = build_model(cfg, dataset.camera_num).cuda()
optimizer = make_optimizer(cfg, model)
scheduler = build_scheduler(optimizer, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.START_ITERS, cfg.SOLVER.END_ITERS, cfg.SOLVER.LR_SCALE)

# Train from checkpoint with fixed iteration number
iter0 = 0
if args.resume != 0:
    iter0 = args.resume
    para_file = os.path.join(cfg.OUTPUT_DIR,'checkpoint_%d.pt' % args.resume)
# or automatically find the maximum one.
elif args.cont:
    para_file, iter0 = get_iteration_path_and_iter(output_dir)
# Resume by the path we found
if args.resume != 0 or args.cont:
    assert os.path.exists(para_file) and para_file != "", 'The checkpoint does not exist:' + para_file
    
    print('Continue training from checkpoint: ',para_file)
    dict_0 = torch.load(os.path.join(output_dir,para_file),map_location='cuda:%d' % int(args.gpu))

    model.load_state_dict(dict_0['model'])
    
    optimizer.load_state_dict(dict_0['optimizer'])
    scheduler.load_state_dict(dict_0['scheduler'])

    logger.info("Load checkpoint:{}  iter:{}".format(output_dir,iter0))
else:
    print('Create a new model for training, dataset:', cfg.DATASETS.TRAIN)

# Set loss function
loss_fn = make_loss(cfg)

# Set psnr threshold to automatically stop the program
psnr_thres=args.psnr_thres

# Train model
do_train(
        cfg,
        model,
        train_loader,
        dataset,
        dataset_val,
        optimizer,
        scheduler,
        loss_fn,
        writer,
        resume_epoch = iter0,
        psnr_thres=psnr_thres
    )
