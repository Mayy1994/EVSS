import os
import sys
import cv2
import numpy as np
sys.path.append('/home/mybeast/xjj/EVSS')

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.flownet import FlowNets
from model.conet import CoNet
from model.warpnet import warp

from model.csrnet_feat import resnet18
from model.csrnet_seg import SemsegModel



from matplotlib import pyplot as plt

class Edge_criterion(nn.Module):
    def __init__(self, ignore_index=255):
        super(Edge_criterion, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, preds, target):

        h, w = target.size(1), target.size(2)

        pos_num = torch.sum(target == 1, dtype=torch.float)
        neg_num = torch.sum(target == 0, dtype=torch.float)

        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = torch.tensor([weight_neg, weight_pos])
        loss = F.cross_entropy(preds, target, weights.cuda(), ignore_index=self.ignore_index)

        return loss

class VSSNet(nn.Module):
    def __init__(self, n_classes=19):
        super(VSSNet, self).__init__()
        self.resnet = resnet18(pretrained=True, efficient=False)
        self.csrnet = SemsegModel(self.resnet, n_classes)
        self.flownet = FlowNets()
        self.conet = CoNet(n_classes=n_classes)
        self.warpnet = warp()
        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=11)
        self.edge_semantic_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.edge_loss = Edge_criterion()

        self.set_fix_csrnet()

    def forward(self, img_list, label=None, edge_label=None):

        n, c, h, w = img_list[0].shape

        img_1_feat = self.csrnet(img_list[0])
        img_1_mask = torch.argmax(img_1_feat, dim=1).unsqueeze(1)


        img_2_mask = self.csrnet(img_list[1])
        img_2_mask = F.upsample(img_2_mask, scale_factor=4, mode='bilinear', align_corners=False)
        img_2_mask = torch.argmax(img_2_mask, dim=1)

        loss_warp = 0.0
        loss_semantic = 0.0
        loss_edge = 0.0
        loss_edge_semantic = 0.0

        # compute optical flow
        flow = self.flownet(torch.cat([img_list[1], img_list[0]], dim=1))

        # the warped features
        feat_warp = self.warpnet(img_1_feat, flow)
        img_2_mask_flow = torch.argmax(feat_warp, dim=1).unsqueeze(1)


        img_2_out_propagate = F.upsample(feat_warp, scale_factor=4, mode='bilinear', align_corners=False)
        
        # warp loss
        loss_warp += self.semantic_loss(img_2_out_propagate, img_2_mask)


        img_2_out_propagate_warp = torch.argmax(img_2_out_propagate, dim=1, keepdims=True).detach()
        label_dm2 = (img_2_out_propagate_warp != img_2_mask.unsqueeze(1)).float().detach()

        # coarse distortion map
        dm_coarse = (img_1_mask != img_2_mask_flow).float().detach()


        # edge & thin objects map; the updated features
        edge, img_2_feat_update = self.conet(img_list[1], feat_warp)
        
        edge = F.upsample(edge, scale_factor=4, mode='bilinear', align_corners=False)
        img_2_feat_update = F.upsample(img_2_feat_update, scale_factor=4, mode='bilinear', align_corners=False)


        # edge loss
        loss_edge += self.edge_loss(edge, edge_label)

        #edge map
        dm_edge = F.softmax(edge, 1)  
        
        dm_coarse = F.upsample(dm_coarse, scale_factor=4, mode='bilinear', align_corners=False)

        # fine distortio map
        dm_fine = 0.8 * dm_edge[:,1,:,:].unsqueeze(1) + 0.2 * dm_coarse

        img_2_out_merge = img_2_out_propagate * (1-dm_fine) + img_2_feat_update * dm_fine


        # semantic loss
        loss_semantic += self.semantic_loss(img_2_out_merge, img_2_mask)   

        edge_argmax = torch.argmax(edge, dim=1).detach()
        img_2_mask_edge = edge_argmax * img_2_mask    
        img_2_feat_update_edge = edge_argmax.unsqueeze(1) * img_2_feat_update
        
        # edge-semantics loss
        loss_edge_semantic += self.edge_semantic_loss(img_2_feat_update_edge, img_2_mask_edge.detach())

        loss_warp = torch.unsqueeze(loss_warp, 0)
        
        loss_semantic = torch.unsqueeze(loss_semantic, 0)

        loss_edge = torch.unsqueeze(loss_edge, 0)

        loss_edge_semantic = torch.unsqueeze(loss_edge_semantic, 0)


        return loss_warp, loss_semantic, loss_edge, loss_edge_semantic

    def set_fix_csrnet(self):
        for param in self.csrnet.parameters():
            param.requires_grad = False




