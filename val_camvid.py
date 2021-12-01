import os
import sys
import cv2
import argparse
import numpy as np
sys.path.append('/home/mybeast/xjj/EVSS/')
os.environ["CUDA_VISIBLE_DEVICES"]="1"



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from model.evss_camvid import VSSNet
from lib.dataset.camvid import camvid_video_dataset_PDA


from matplotlib import pyplot as plt
from lib.dataset.utils import runningScore



def get_arguments():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--data_list_path", type=str, help="path to the data list")
    parser.add_argument("--data_path", type=str, help="path to the data")
    parser.add_argument("--gt_path", type=str, help="path to the ground truth")
    parser.add_argument("--evss_model", type=str, help="path to the trained evss model")
    parser.add_argument("--wf_val", type=float, default=0.2, help="hyper parameter")
    parser.add_argument("--num_workers", type=int, help="num of cpus used")
    

    return parser.parse_args()


def test():
    args = get_arguments()
    print(args)

    net = VSSNet(n_classes=11)
    old_weight = torch.load(args.evss_model)
    new_weight = {}
    for k, v in old_weight.items():
        new_k = k.replace('module.', '')
        new_weight[new_k] = v
    
    net.load_state_dict(new_weight, strict=True)
    net.cuda().eval()

    csrnet = net.csrnet
    flownet = net.flownet
    conet = net.conet
    warpnet = net.warpnet

    test_data = camvid_video_dataset_PDA(args.data_path, args.gt_path, args.data_list_path)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)

    distance = 10
    wf = args.wf_val
    miou_cal = runningScore(n_classes=11)
    
    
    with torch.no_grad():
        # for d in range(1, distance):
        for d in [3,5,7,9]:
        
            for step, sample in enumerate(test_data_loader):

                img_list, gt_label = sample
                img = img_list[9 - d].cuda()
                feat = csrnet(img)
                img_1_mask = torch.argmax(feat, dim=1).unsqueeze(1)
                
        
                for i in range(d):
                    img_1 = img_list[9 - d + i].cuda()
                    img_2 = img_list[10 - d + i].cuda()
                    flow = flownet(torch.cat([img_2, img_1], dim=1))
                    feat = warpnet(feat, flow)
                    
                img_2_mask_flow = torch.argmax(feat, dim=1).unsqueeze(1)
                dm_coarse = (img_1_mask - img_2_mask_flow)
                dm_coarse[dm_coarse != 0] = 1
                dm_coarse = dm_coarse.float()

                dm_coarse = F.upsample(dm_coarse, scale_factor=4, mode='bilinear', align_corners=False)

                
                edge, feat_update = conet(img_2, feat)
                feat_update_up = F.interpolate(feat_update, scale_factor=4, mode='bilinear', align_corners=False)
                edge = F.interpolate(edge, scale_factor=4, mode='bilinear', align_corners=False)


                feat_warp_up = F.interpolate(feat, scale_factor=4, mode='bilinear', align_corners=False)
                dm_edge = F.softmax(edge, 1)

                dm_fine = (1-wf) * dm_edge[:,1,:,:].unsqueeze(1) + wf * dm_coarse
                

                feat_merge = feat_warp_up * (1-dm_fine) + feat_update_up * dm_fine
                
                feat_merge_down = F.interpolate(feat_merge, scale_factor=0.25, mode='bilinear', align_corners=False)
                img_1_mask = torch.argmax(feat_merge_down, dim=1).unsqueeze(1)
                

                out = torch.argmax(feat_merge, dim=1)
                out = out.squeeze().cpu().numpy()
                gt_label_miou = gt_label.squeeze().cpu().numpy()
                miou_cal.update(gt_label_miou, out)
                

            miou, iou = miou_cal.get_scores(return_class=True)
            miou_cal.reset()
            print('distance:{} miou:{}'.format(d, miou))
            print('class iou:')
            for i in range(len(iou)):
                print(iou[i])



if __name__ == '__main__':
    test()
