# Maybe useless
import os
import sys

sys.path.append('..')
from config import cfg
from data import make_ray_data_loader, make_ray_data_loader_view, get_iteration_path_and_iter
from engine.layered_trainer import do_evaluate
from modeling import build_layered_model
from solver import make_optimizer, WarmupMultiStepLR, build_scheduler
from layers import make_loss

from utils.logger import setup_logger
from layers.RaySamplePoint import RaySamplePoint

from torch.utils.tensorboard import SummaryWriter
import torch

import argparse

# Parse arguments
text = 'This is the program to train the nerf by all frames and layers, try to get help by using -h'
parser = argparse.ArgumentParser(description=text)
parser.add_argument('-c', '--config', default='', help='set the config file path to train the network')
parser.add_argument('-g','--gpu', type=int, default=0, help='set gpu id to train the network')
parser.add_argument('-r','--resume', type=int, default = 0, help='set the checkpoint number to resume training')
parser.add_argument('-s','--psnr_thres', type=float, default = 100.0, help= 'set the psnr threshold to train next frame')
parser.add_argument('-cont','--cont',action="store_true", help='render all layers')
args = parser.parse_args()

# Set PyTorch GPU id and settings
torch.cuda.set_device(args.gpu)
torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)

# Load config
training_config = args.config
assert os.path.exists(training_config), 'training config does not exist.'
cfg.merge_from_file(training_config)
cfg.freeze()

# Initialize writer and logger
output_dir = cfg.OUTPUT_DIR
writer = SummaryWriter(log_dir=output_dir)
writer.add_text('OUT_PATH', output_dir,0)
logger = setup_logger("RFRender", output_dir, 0)
logger.info("Running with config:\n{}".format(cfg))

# Create ray dataset
train_loader, dataset = make_ray_data_loader(cfg)
val_loader, dataset_val = make_ray_data_loader_view(cfg)

# Create model, optimizer and scheduler
model = build_layered_model(cfg, dataset.camera_num).cuda()
model.set_bkgd_bbox(dataset.datasets[0][0].layer_bbox)
model.set_bboxes(dataset.bboxes)
model_dict = model.state_dict()
# for k,v in model_dict.items():
#     print(k)
optimizer = make_optimizer(cfg, model)
scheduler = build_scheduler(optimizer, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.START_ITERS, cfg.SOLVER.END_ITERS, cfg.SOLVER.LR_SCALE)

# Train from checkpoint with fixed iteration number or automatically find the maximum one.
cont = args.resume != 0 or args.cont
iter0 = 0
if cont:
    if args.resume != 0:
        iter0 = args.resume
        para_file = 'layered_rfnr_checkpoint_%d.pt' % args.resume
    else:
        para_file, iter0 = get_iteration_path_and_iter(output_dir)
        
    assert os.path.exists(para_file) and para_file != "", 'The checkpoint does not exist'
    print('Continue training from checkpoint: ',para_file)
    dict_0 = torch.load(os.path.join(output_dir,para_file),map_location='cuda:%d' % int(args.gpu))
    # Add non-existing parameters into pretrained model parameters
    model_dict = dict_0['model']
    model_new_dict = model.state_dict()
    offset = {k: v for k, v in model_new_dict.items() if k not in model_dict}
    for k,v in offset.items():
        model_dict[k] = v

    model.load_state_dict(model_dict)
    
    optimizer.load_state_dict(dict_0['optimizer'])
    scheduler.load_state_dict(dict_0['scheduler'])

    logger.info("Load checkpoint:{}  iter:{}".format(output_dir,iter0))
else:
    print('Create a new model for training ', cfg.DATASETS.TRAIN)

loss_fn = make_loss(cfg)
psnr_thres=args.psnr_thres

#TODO: Modify to layered space-time pipeline
do_evaluate(model, dataset_val)
