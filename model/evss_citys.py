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
        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.edge_semantic_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.edge_loss = Edge_criterion()

        self.set_fix_csrnet()

    def forward(self, img_list, label=None, edge_label=None):

        n, c, h, w = img_list[0].shape

        img_1_feat = self.csrnet(img_list[0])
        warp_img = F.upsample(img_list[0], scale_factor=0.25, mode='bilinear', align_corners=False)
        img_1_mask = torch.argmax(img_1_feat, dim=1).unsqueeze(1)


        img_2_mask = self.csrnet(img_list[1])
        img_2_mask = F.upsample(img_2_mask, scale_factor=4, mode='bilinear', align_corners=False)
        img_2_mask = torch.argmax(img_2_mask, dim=1)

        loss_warp = 0.0
        loss_semantic = 0.0
        loss_edge = 0.0
        loss_edge_semantic = 0.0

        flow = self.flownet(torch.cat([img_list[1], img_list[0]], dim=1))
        

        img_2_feat = self.warpnet(img_1_feat, flow)
        img_2_mask_flow = torch.argmax(img_2_feat, dim=1).unsqueeze(1)

        warp_img = self.warpnet(warp_img, flow)

        # semantic loss
        img_2_out_propagate = F.upsample(img_2_feat, scale_factor=4, mode='bilinear', align_corners=False)
        loss_warp += self.semantic_loss(img_2_out_propagate, img_2_mask)


        img_2_out_propagate_warp = torch.argmax(img_2_out_propagate, dim=1, keepdims=True).detach()
        label_dm2 = (img_2_out_propagate_warp != img_2_mask.unsqueeze(1)).float().detach()

        dm_flow = (img_1_mask != img_2_mask_flow).float().detach()


        edge, img_2_feat_cc = self.conet(img_list[1], img_2_feat)
        edge = F.upsample(edge, scale_factor=4, mode='bilinear', align_corners=False)
        img_2_out_cc = F.upsample(img_2_feat_cc, scale_factor=4, mode='bilinear', align_corners=False)


        loss_edge += self.edge_loss(edge, edge_label)

        dm_2 = F.softmax(edge, 1)
        dm_flow = F.upsample(dm_flow, scale_factor=4, mode='bilinear', align_corners=False)

        dm_2 = 0.5 * dm_2[:,1,:,:].unsqueeze(1) + 0.5 * dm_flow

        img_2_out_merge = img_2_out_propagate * (1-dm_2) + img_2_out_cc*dm_2

        loss_semantic += self.semantic_loss(img_2_out_merge, img_2_mask)

        edge_argmax = torch.argmax(edge, dim=1).detach()
        img_2_mask_edge = edge_argmax * img_2_mask
        img_2_out_cc_edge = edge_argmax.unsqueeze(1) * img_2_out_cc
        loss_edge_semantic += self.edge_semantic_loss(img_2_out_cc_edge, img_2_mask_edge.detach())


        loss_warp /= 2
        loss_warp = torch.unsqueeze(loss_warp, 0)
        
        loss_semantic /= 2
        loss_semantic = torch.unsqueeze(loss_semantic, 0)


        loss_edge /= 2
        loss_edge = torch.unsqueeze(loss_edge, 0)

        loss_edge_semantic /= 2
        loss_edge_semantic = torch.unsqueeze(loss_edge_semantic, 0)


        return loss_warp, loss_semantic, loss_edge, loss_edge_semantic

    def set_fix_csrnet(self):
        for param in self.csrnet.parameters():
            param.requires_grad = False



