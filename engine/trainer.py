# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
logging.basicConfig(level = logging.INFO)
import imageio
import torch

from utils import batchify_ray, vis_density, metrics
from utils.metrics import *
import numpy as np
import os
import time
import math


def evaluator(val_dataset, model, loss_fn, swriter, epoch):
    
    model.eval()
    rays, colors, image, near_far = val_dataset[0]

    color_gt = image.cuda()

    rays = rays.cuda()
    colors = colors.cuda()
    #bbox = bbox.cuda()
    near_far = near_far.cuda()

    with torch.no_grad():
        results = batchify_ray(model, rays, near_far=near_far)
        H, W, C = color_gt.shape
        color_img = results['fine_color'].reshape(H, W, C)
        depth_img = results['fine_depth'].reshape(H, W, 1)
        acc_map = results['fine_acc'].reshape(H, W, 1)

        color_img_0 = results['coarse_color'].reshape(H, W, C)
        depth_img_0 = results['coarse_depth'].reshape(H, W, 1)
        acc_map_0 = results['coarse_acc'].reshape(H, W, 1)
        
        swriter.add_image('GT/Image', color_gt.permute(2,0,1), epoch)

        swriter.add_image('fine/rendered', color_img.permute(2,0,1), epoch)
        swriter.add_image('fine/depth', depth_img.permute(2,0,1), epoch)
        swriter.add_image('fine/alpha', acc_map.permute(2,0,1), epoch)

        swriter.add_image('coarse/rendered', color_img_0.permute(2,0,1), epoch)
        swriter.add_image('coarse/depth', depth_img_0.permute(2,0,1), epoch)
        swriter.add_image('coarse/alpha', acc_map_0.permute(2,0,1), epoch)

        loss_map = (color_gt-color_img) ** 2
        loss_map = torch.mean(loss_map, dim=2, keepdim=True)

        loss_map[~results['ray_mask'][..., 0].reshape(loss_map.size(0),loss_map.size(1),loss_map.size(2))] = 0
        loss_map = loss_map / (loss_map.max() - loss_map.min())
        swriter.add_image('fine/loss_map', loss_map.permute(2,0,1), epoch)

        return loss_fn(color_img, color_gt).item()


def do_train(
        cfg,
        model,
        train_loader,
        train_dataset,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        swriter,
        resume_epoch = 0,
        psnr_thres = 100
):

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    max_epochs = cfg.SOLVER.MAX_EPOCHS
    coarse_stage = cfg.SOLVER.COARSE_STAGE

    logger = logging.getLogger("NeRF.%s.train" % cfg.OUTPUT_DIR.split('/')[-1]) 
    logger.setLevel(logging.DEBUG)
    logger.info("Start training")
    #global step
    global_step = 0
    
    torch.autograd.set_detect_anomaly(True)


    for epoch in range(1+resume_epoch,max_epochs):
        print('Training Epoch %d...' % epoch)
        model.cuda()

        #psnr monitor 
        psnr_monitor = []

        #epoch time recordingbatchify_ray
        epoch_start = time.time()
        for batch_idx, batch in enumerate(train_loader):

            #iteration time recording
            iters_start = time.time()
            global_step = (epoch - 1) * len(train_loader) + batch_idx

            model.train()
            optimizer.zero_grad()

            rays, colors, near_far = batch 

            rays = rays.cuda()
            colors = colors.cuda()
            #bboxes = bboxes.cuda()
            near_far = near_far.cuda()

            loss = 0
           
            results = model(rays, near_far=near_far)
            ray_mask = results['ray_mask']

            coarse_color = results['coarse_color'][ray_mask]
            fine_color = results['fine_color'][ray_mask]

            colors = colors[ray_mask]

            loss1 = loss_fn(coarse_color, colors)
            loss2 = loss_fn(fine_color, colors)

            loss = loss + loss1 + loss2

            loss.backward()

            optimizer.step()
            scheduler.step()

            psnr_0 = psnr(coarse_color, colors)
            psnr_ = psnr(fine_color, colors)
            psnr_monitor.append(psnr_.cpu().detach().numpy())


            if batch_idx % 50 ==0:
                swriter.add_scalar('Loss/train_loss',loss.item(), global_step)
                swriter.add_scalar('TrainPsnr', psnr_, global_step)

            if batch_idx % log_period == 0:
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3e}  Psnr coarse: {:.2f} Psnr fine: {:.2f} Lr: {:.2e} Speed: {:.1f}[rays/s]"
                            .format(epoch, batch_idx, len(train_loader), loss.item(), psnr_0, psnr_ ,lr,
                                    log_period * float(cfg.SOLVER.BUNCH) / (time.time() - iters_start)))
            #validation
            if (global_step+log_period) % 1000 == 0:
                val_vis(val_loader, model, loss_fn, swriter, logger, epoch)

            #model saving
            if global_step % checkpoint_period == 0:
                ModelCheckpoint(model, optimizer, scheduler, output_dir, epoch)
                    
        #EPOCH COMPLETED
        ModelCheckpoint(model, optimizer, scheduler, output_dir, epoch)

        val_vis(val_loader, model ,loss_fn, swriter, logger, epoch)

        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[rays/s]'
                    .format(epoch, time.time() - epoch_start,
                            len(train_loader) * float(cfg.SOLVER.BUNCH) / (time.time() - epoch_start)))

        psnr_monitor = np.mean(psnr_monitor)
        
        if psnr_monitor > psnr_thres:
            logger.info("The Mean Psnr of Epoch: {:.3f}, greater than threshold: {:.3f}, Training Stopped".format(psnr_monitor, psnr_thres))
            break
        else:
            logger.info("The Mean Psnr of Epoch: {:.3f}, less than threshold: {:.3f}, Continue to Training".format(psnr_monitor, psnr_thres))

def val_vis(val_loader,model ,loss_fn, swriter, logger, epoch):
   
    avg_loss = evaluator(val_loader, model, loss_fn, swriter,epoch)
    logger.info("Validation Results - Epoch: {} Avg Loss: {:.3f}"
                .format(epoch,  avg_loss)
                )
    swriter.add_scalar('Loss/val_loss',avg_loss, epoch)

def ModelCheckpoint(model, optimizer, scheduler, output_dir, epoch):
    # model,optimizer,scheduler saving 
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()}, 
                os.path.join(output_dir,'checkpoint_%d.pt' % epoch))
        




        

