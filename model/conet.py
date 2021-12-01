import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)  

class PSPModule(nn.Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=128, sizes=(1, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.SyncBatchNorm(out_features),
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = nn.SyncBatchNorm(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [ F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class Upsample_sk(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3, separable=False):
        super(Upsample_sk, self).__init__()
        self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn, separable=separable)
        self.upsampling_method = upsample
        self.sk = SK_Block(num_maps_in, num_maps_in)

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        skip_size = skip.size()[2:4]
        x = self.upsampling_method(x, skip_size)
        x = self.sk(x, skip)
        x = self.blend_conv.forward(x)
        return x

class _BNReluConv(nn.Sequential):

    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1,
                 drop_rate=.0, separable=False):
        super(_BNReluConv, self).__init__()
        padding = k // 2
        conv_class = SeparableConv2d if separable else nn.Conv2d
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, padding=padding, bias=bias, dilation=dilation))
        if batch_norm:
            self.add_module('norm', nn.SyncBatchNorm(num_maps_out))
        self.add_module('relu', nn.LeakyReLU(0.1, inplace=True))
    
        if drop_rate > 0:
            self.add_module('dropout', nn.Dropout2d(drop_rate, inplace=True))

class SK_Block(nn.Module):
    def __init__(self, in_chan, mid_chan, *args, **kwargs):
        super(SK_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _BNReluConv(in_chan, mid_chan, k=1, batch_norm=True, separable=False)
        self.fc1 = nn.Linear(mid_chan, in_chan)
        self.fc2 = nn.Linear(mid_chan, in_chan)
        self.softmax = nn.Softmax(dim=1)
        self.conv_out = _BNReluConv(in_chan, in_chan, k=1, batch_norm=True, separable=False)



    def forward(self, feat1, feat2):
        feats = feat1 + feat2
        feat_s = self.avg_pool(feats)
        feat_z = self.fc(feat_s).squeeze()
        feat_1 = self.fc1(feat_z)
        feat_2 = self.fc2(feat_z)
        if len(feat_2.shape)== 1:
            feat_1 = feat_1.unsqueeze(0)
            feat_2 = feat_2.unsqueeze(0)

        feat_1_2 = torch.stack([feat_1, feat_2], 1).unsqueeze(-1)
        feat_1_2 = self.softmax(feat_1_2)
        feat_1 = feat_1_2[:,0,:,:].unsqueeze(-1)
        feat_2 = feat_1_2[:,1,:,:].unsqueeze(-1)
        feat1_new = feat1 * feat_1.expand_as(feat1)
        feat2_new = feat2 * feat_2.expand_as(feat2)
        feat_sum = self.conv_out(feat1_new + feat2_new)

        return feat_sum

class Edge_Module(nn.Module):

    def __init__(self,in_fea=[256,512,1024], mid_fea=256, out_fea=11):
        super(Edge_Module, self).__init__()
        
        self.conv1 =  nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.SyncBatchNorm(mid_fea)
            ) 
        self.conv2 =  nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.SyncBatchNorm(mid_fea)
            )  
        self.conv3 =  nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.SyncBatchNorm(mid_fea)
        )
        self.conv4 = nn.Conv2d(mid_fea,out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea*3,out_fea, kernel_size=1, padding=0, dilation=1, bias=True)

            

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()
        
        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)        
        
        
        edge2 =  F.interpolate(edge2, size=(h, w), mode='bilinear',align_corners=False)
        edge3 =  F.interpolate(edge3, size=(h, w), mode='bilinear',align_corners=False) 
 
        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge = self.conv5(edge)

         
        return edge

class CoNet(nn.Module):
    def __init__(self, in_planes=3, n_classes=19):

        super(CoNet, self).__init__()

        self.num_features = 128

        self.conv1 = self.conv(in_planes, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = self.conv(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = self.conv(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3_1 = self.conv(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = self.conv(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = self.conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = self.conv(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = self.conv(512, 512, kernel_size=3, stride=1, padding=1)

        self.deconv4 = Upsample_sk(self.num_features, 256, self.num_features)
        self.deconv3 = Upsample_sk(self.num_features, 128, self.num_features // 2)
        self.deconv2 = Upsample_sk(self.num_features // 2, 64, self.num_features // 4)

        self.predict_cc = nn.Conv2d(self.num_features // 4 + n_classes, n_classes, 1, stride=1, padding=0)


        self.psp = PSPModule(512,self.num_features)

        self.edge_layer = Edge_Module(in_fea=[32, 64, 128], mid_fea=32, out_fea = 2)


    def forward(self, image, feat_warp):
        mean = torch.tensor((0.485, 0.456, 0.406)).cuda()
        std = torch.tensor((0.229, 0.224, 0.225)).cuda()
        image = image.sub(mean[None,:,None,None]).div(std[None,:,None,None])
        
        out_conv1 = self.conv1(image)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))


        out_conv5 = self.psp(out_conv5)

        out_deconv4 = self.deconv4(out_conv5, out_conv4)
        out_deconv3 = self.deconv3(out_deconv4, out_conv3)
        out_deconv2 = self.deconv2(out_deconv3, out_conv2)

        edge = self.edge_layer(out_deconv2, out_deconv3, out_deconv4)


        feat = torch.cat((out_deconv2, feat_warp), 1)
        correction_cue = self.predict_cc(feat)


        return edge, correction_cue

    def conv(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.SyncBatchNorm(out_planes), nn.LeakyReLU(0.1, inplace=True))

