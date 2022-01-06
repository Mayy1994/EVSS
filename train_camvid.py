import os
import sys
import ast
import random
import argparse
import numpy as np
sys.path.append('/home/mybeast/xjj/EVSS')


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tensorboardX import SummaryWriter

from lib.dataset.camvid import camvid_video_dataset, camvid_video_dataset_PDA
from lib.dataset.utils import runningScore
from model.evss_camvid import VSSNet

import time
import datetime


def get_arguments():
    parser = argparse.ArgumentParser(description="Train EVSS")
    parser.add_argument("--exp_name", type=str, help="exp name")
    parser.add_argument("--root_data_path", type=str, help="root path to the dataset")
    parser.add_argument("--root_gt_path", type=str, help="root path to the ground truth")
    parser.add_argument("--train_list_path", type=str, help="path to the list of train subset")
    parser.add_argument("--test_list_path", type=str, help="path to the list of test subset")
    parser.add_argument("--model_name", type=str, help="name for the training model")
    parser.add_argument("--resume", type=ast.literal_eval, default=False, help="resume or not")
    parser.add_argument("--resume_epoch", type=int, help="from which epoch for resume")
    parser.add_argument("--resume_load_path", type=str, help="resume model load path")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--local_rank", type=int, help="index the replica")
    parser.add_argument("--conet_lr", type=float, help="learning rate of conet")
    parser.add_argument("--wf", type=float, default=0.5, help="learning rate of conet")
    parser.add_argument("--random_seed", type=int, help="random seed")
    parser.add_argument("--train_flownet", type=ast.literal_eval, default=True, help="trian flownet or not")
    parser.add_argument("--train_power", type=float, help="power value for linear learning rate schedule")
    parser.add_argument("--final_lr", type=float, default=1e-6, help="learning rate in the second stage")
    parser.add_argument("--weight_decay", type=float, help="learning rate")
    parser.add_argument("--train_batch_size", type=int, help="train batch size")
    parser.add_argument("--train_shuffle", type=ast.literal_eval, default=True, help="shuffle or not in training")
    parser.add_argument("--train_num_workers", type=int, default=8, help="num cpu use")
    parser.add_argument("--num_epoch", type=int, default=100, help="num of epoch in training")
    parser.add_argument("--snap_shot", type=int, default=1, help="save model every per snap_shot")
    parser.add_argument("--model_save_path", type=str, help="model save path")
    parser.add_argument("--test_batch_size", type=int, default=1, help="batch_size for validation")
    parser.add_argument("--test_shuffle", type=ast.literal_eval, default=False, help="shuffle or not in validation")
    parser.add_argument("--test_num_workers", type=int, default=4, help="num of used cpus in validation")
    parser.add_argument("--use_tensorboard", type=ast.literal_eval, default=True, help="use tensorboard or not")
    parser.add_argument("--tblog_dir", type=str, help="log save path")
    parser.add_argument("--tblog_interval", type=int, default=50, help="interval for tensorboard logging")

    return parser.parse_args()


def make_dirs(args):
    if args.use_tensorboard and not os.path.exists(args.tblog_dir):
        os.makedirs(args.tblog_dir)
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)


def train():
    torch.distributed.init_process_group(backend="nccl")

    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    args = get_arguments()
    if local_rank == 0:
        print(args)
        make_dirs(args)

    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if local_rank == 0:
        print('random seed:{}'.format(random_seed))

    if local_rank == 0 and args.use_tensorboard:
        tblogger = SummaryWriter(args.tblog_dir)

    net = VSSNet(n_classes=11)

    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    if args.resume:
        old_weight = torch.load(args.resume_load_path, map_location=map_location)
        new_weight = {}
        for k, v in old_weight.items():
            k = k.replace('module.', '')
            new_weight[k] = v
        net.load_state_dict(new_weight, strict=True)
        
        start_epoch = args.resume_epoch
    else:
        # load the backbone network and flownet
        csrnet_weight = torch.load('/home/mybeast/xjj/EVSS/saved_model/pretrained/csrnet_camvid_best.pth', map_location=map_location)
        net.csrnet.load_state_dict(csrnet_weight, True)
        
        flownet_weight = torch.load('/home/mybeast/xjj/EVSS/saved_model/pretrained/flownet.pth')
        net.flownet.load_state_dict(flownet_weight, True)
        
        start_epoch = 0
        

    if local_rank == 0:
        print('Successful loading model!')

    net.cuda()
    net = nn.parallel.DistributedDataParallel(net,
                                              device_ids=[local_rank],
                                              output_device=local_rank,
                                              find_unused_parameters=True)

    # prepare the training data and testing data
    train_data = camvid_video_dataset(args.root_data_path,
                                      args.root_gt_path,
                                      args.train_list_path,
                                      crop_size=(480, 720))

    train_data_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=args.train_batch_size,
                                                    shuffle=False,
                                                    pin_memory=False,
                                                    num_workers=args.train_num_workers,
                                                    drop_last=True,
                                                    sampler=DistributedSampler(train_data,
                                                                               num_replicas=world_size,
                                                                               rank=local_rank,
                                                                               shuffle=True))

    test_data = camvid_video_dataset_PDA(args.root_data_path, args.root_gt_path, args.test_list_path)
    test_data_loader = torch.utils.data.DataLoader(test_data,
                                                   batch_size=args.test_batch_size,
                                                   shuffle=args.test_shuffle,
                                                   num_workers=args.test_num_workers)
    # optimizer
    flow_params = []
    for m in net.module.flownet.modules():
        for p in m.parameters():
            flow_params.append(p)
    flow_optimizer = optim.Adam(params=flow_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0)
    flow_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(flow_optimizer, args.num_epoch, args.final_lr)

    conet_params = []
    for m in net.module.conet.modules():
        for p in m.parameters():
            conet_params.append(p)
    conet_optimizer = optim.Adam(params=conet_params, lr=args.conet_lr, betas=(0.9, 0.999), weight_decay=0)
    conet_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(conet_optimizer, args.num_epoch, args.final_lr)

    
    miou_cal = runningScore(n_classes=11)
    current_miou = 0
    wf = args.wf
    itr = start_epoch * len(train_data_loader)
    max_itr = args.num_epoch * len(train_data_loader)

    st = glob_st = time.time()

    # begin training
    for epoch in range(start_epoch, args.num_epoch):

        net.module.csrnet.eval()
        net.module.flownet.train()
        net.module.conet.train()
        flow_lr_scheduler.step()
        conet_lr_scheduler.step()

        train_data_loader.sampler.set_epoch(epoch)


        for i, data_batch in enumerate(train_data_loader):
            img_list, gt_label, edge_label = data_batch

            # four kinds of losses
            loss_warp, loss_semantic, loss_edge, loss_edge_semantic = net(img_list, label=gt_label, edge_label = edge_label)
            loss_warp = torch.mean(loss_warp)
            loss_semantic = torch.mean(loss_semantic)
            loss_edge = torch.mean(loss_edge)
            loss_edge_semantic = torch.mean(loss_edge_semantic)

            loss = 0.2 * loss_warp + 0.5 * loss_semantic  + 0.1 * loss_edge + 0.2 * loss_edge_semantic

            loss.backward()

            flow_optimizer.step()
            conet_optimizer.step()

            flow_optimizer.zero_grad()
            conet_optimizer.zero_grad()


            itr += 1

            if local_rank == 0:
                ed = time.time()
                t_intv, glob_t_intv = ed - st, ed - glob_st
                eta = int((max_itr - itr) * (glob_t_intv / itr))
                eta = str(datetime.timedelta(seconds=eta))
                print('epoch:{}/{} batch:{}/{} iter:{} loss_warp:{:05f} loss_semantic:{:05f} loss_edge:{:05f} loss_edge_semantic:{:05f} eta:{}'.format(
                    epoch+1, args.num_epoch, i, len(train_data_loader), itr, loss_warp.item(), loss_semantic.item(), loss_edge.item(), loss_edge_semantic.item(), eta))

                if args.use_tensorboard and itr % args.tblog_interval == 0:
                    tblogger.add_scalar('loss_warp', loss_warp.item(), itr)
                    tblogger.add_scalar('loss_semantic', loss_semantic.item(), itr)
                    tblogger.add_scalar('loss_edge', loss_edge.item(), itr)
                    tblogger.add_scalar('loss_edge_semantic', loss_edge_semantic.item(), itr)

                st = ed

        
        dist.barrier()

        if (epoch+1) % args.snap_shot == 0 or epoch == 0:
            net.eval()
            distance_list = [1, 5, 9]
            miou_list, iou_list = [], []
            for d in distance_list:
                with torch.no_grad():
                    for step, sample in enumerate(test_data_loader):
                       
                        img_list, gt_label = sample
                        gt_label = gt_label.squeeze().cpu().numpy()

                        img = img_list[9 - d].cuda()
                        feat = net.module.csrnet(img)
                        img_1_mask = torch.argmax(feat, dim=1).unsqueeze(1)
                        
                        
                        for i in range(d):
                            img_1 = img_list[9 - d + i].cuda()
                            img_2 = img_list[10 - d + i].cuda()
                            flow = net.module.flownet(torch.cat([img_2, img_1], dim=1))
                            # the warped features
                            feat = net.module.warpnet(feat, flow)
                            
                        
                        img_2_mask_flow = torch.argmax(feat, dim=1).unsqueeze(1)
                        
                        # coarse distortion map
                        dm_coarse = (img_1_mask - img_2_mask_flow)
                        dm_coarse[dm_coarse != 0] = 1
                        dm_coarse = dm_coarse.float()
                        dm_coarse = F.upsample(dm_coarse, scale_factor=4, mode='bilinear', align_corners=False)

                        # edge map; the updated features
                        edge, feat_update = net.module.conet(img_2, feat)
                        feat_update_up = F.interpolate(feat_update, scale_factor=4, mode='bilinear', align_corners=False)
                        feat_warp_up = F.interpolate(feat, scale_factor=4, mode='bilinear', align_corners=False)
                        
                        # edge map
                        edge = F.interpolate(edge, scale_factor=4, mode='bilinear', align_corners=False)
                        dm_edge = F.softmax(edge, 1)

                        # fine distortion map
                        dm_fine = (1-wf) * dm_edge[:,1,:,:].unsqueeze(1) + wf * dm_coarse

                        # the merged features
                        feat_merge = feat_warp_up * (1-dm_fine) + feat_update_up * dm_fine
                        
                        out = torch.argmax(feat_merge, dim=1)
                        out = out.squeeze().cpu().numpy()
                        miou_cal.update(gt_label, out)

                        

                miou, iou = miou_cal.get_scores(return_class=True)
                miou_cal.reset()
                miou_list.append(miou)
                iou_list.append(iou)

                if local_rank == 0:
                    print('distance:{} miou:{}'.format(d, miou))
                    print('class iou:')
                    for i in range(len(iou)):
                        print(iou[i])

            if local_rank == 0:
                if args.use_tensorboard:
                    for i, d in enumerate(distance_list):
                        tblogger.add_scalar('miou/distance_{}'.format(d), miou_list[i], epoch)

                save_name = 'now.pth'
                save_path = os.path.join(args.model_save_path, save_name)
                torch.save(net.state_dict(), save_path)

                miou = np.mean(miou_list)
                if miou > current_miou:
                    save_name = 'best.pth'
                    save_path = os.path.join(args.model_save_path, save_name)
                    torch.save(net.state_dict(), save_path)
                    current_miou = miou

            dist.barrier()

    if local_rank == 0:
        save_name = 'final.pth'
        save_path = os.path.join(args.model_save_path, save_name)
        torch.save(net.state_dict(), save_path)
        print('%s has been saved' % save_path)


def adjust_lr(args, optimizer, itr, max_itr, lr):
    if itr > max_itr / 2:
        now_lr = lr / 10
    else:
        now_lr = lr

    for group in optimizer.param_groups:
        group['lr'] = now_lr


if __name__ == '__main__':
    train()
    dist.destroy_process_group()
