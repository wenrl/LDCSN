import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import pdb

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Norm(nn.Module):
    def __init__(self, name, n_feats):
        super(Norm, self).__init__()
        assert name in ['bn', 'gn', 'gbn', 'none']
        if name == 'bn':
            self.norm = nn.BatchNorm2d(n_feats)
        elif name == 'gn':
            self.norm = nn.GroupNorm(32, n_feats)
        elif name == 'gbn':
            self.norm = nn.Sequential(nn.GroupNorm(32, n_feats, affine=False),nn.BatchNorm2d(n_feats))
        elif name == 'none':
            pass
        self.name = name

    def forward(self, x):
        if self.name == 'none':
            return x
        else:
            return self.norm(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BamSpatialAttention(nn.Module):
    def __init__(self,channel,reduction = 16, dilation_ratio =2):
        super(BamSpatialAttention,self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1),

            nn.BatchNorm2d(channel//reduction),
            nn.Conv2d(channel//reduction,channel//reduction,3,padding=dilation_ratio,dilation=dilation_ratio),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(True),

            nn.BatchNorm2d(channel // reduction),
            nn.Conv2d(channel // reduction, channel // reduction, 3, padding=dilation_ratio, dilation=dilation_ratio),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(True),

            nn.Conv2d(channel//reduction,1,1)
        )
    def forward(self, x):
        return self.body(x).expand_as(x)


class BamChannelAttention(nn.Module):
    def __init__(self,channel,reduction = 16):
        super(BamChannelAttention,self).__init__()
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1),
            #nn.BatchNorm2d(channel//reduction)
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction,channel,1),
        )
    def forward(self,x):
        out = self.avgPool(x)
        out = self.fc(out)
        return out.expand_as(x)

class BAM(nn.Module):
    def __init__(self, channel, att = 'both', reduction=16):
        super(BAM, self).__init__()
        self.att = att
        self.channelAtt =None
        self.spatialAtt =None
        if att == 'both' or att == 'c':
            self.channelAtt = BamChannelAttention(channel,reduction)
        if att == 'both' or att == 's':
            self.spatialAtt = BamSpatialAttention(channel,reduction)

    def forward(self, x):
        if self.att =='both':
            y1 = self.spatialAtt(x)
            y2 = self.channelAtt(x)
            y = y1+ y2
        elif self.att =='c':
            y = self.channelAtt(x)
        elif self.att =='s':
            y = self.spatialAtt(x)
        return (1 +F.sigmoid(y))*x
