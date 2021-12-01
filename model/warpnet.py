import torch
import torch.nn as nn

from model.resample2d_package.resample2d import Resample2d
import sys
sys.path.append('/home/mybeast/xjj/EVSS')


class warp(nn.Module):

    def __init__(self):
        super(warp, self).__init__()
        self.resample = Resample2d()

    def forward(self, feature, flow):
        # import pdb; pdb.set_trace()
        assert feature.shape[2:] == flow.shape[2:]
        out = self.resample(feature, flow)
        return out
