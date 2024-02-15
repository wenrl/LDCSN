import torch
import torch.nn as nn
from models.SENet import Se
from models.ECBAM import *
from Trans_cnn.conformer import F2S,S2F,Block_unit,Block_unit_MHSA

class feauture_Reduction(nn.Module):
    def __init__(self, inplane, outplane,ufa=True):
        super(feauture_Reduction, self).__init__()
        self.DWconv = nn.Conv2d(inplane, outplane, kernel_size=3, stride=2, padding=1, groups=inplane)
        self.PWconv = nn.Conv2d(outplane, outplane, kernel_size=1, stride=1, padding=0)
        self.PWconv1 = nn.Conv2d(inplane, outplane, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(outplane)
        self.bn1 = nn.BatchNorm2d(outplane)
        self.act_layer1 = nn.Identity()
        if ufa:
            if inplane == 256:
                self.ds = nn.AvgPool2d(2, 2, padding=(1, 0))
            else:
                self.ds = nn.AvgPool2d(2, 2, padding=0)
            self.act_layer = nn.PReLU(outplane)
        else:
            self.ds = nn.AvgPool2d(2, 2, padding=0)
            self.act_layer = nn.Identity()

    def forward(self, x):
        x = self.bn1(self.act_layer1(self.PWconv1(self.ds(x)))) + self.bn(self.act_layer(self.PWconv(self.DWconv(x))))
        return x

# class block_Reduction(nn.Module):
#     def __init__(self, inplane, outplane):
#         super(block_Reduction, self).__init__()
#         self.DWconv = nn.Conv2d(inplane, outplane, kernel_size=3, stride=2, padding=1, groups=inplane)
#         self.PWconv = nn.Conv2d(outplane, outplane, kernel_size=1, stride=1, padding=0)
#         self.PWconv1 = nn.Conv2d(inplane, outplane, kernel_size=1, stride=1, padding=0)
#         self.act_layer = nn.PReLU(outplane)
#         self.bn = nn.BatchNorm2d(outplane)
#         self.bn1 = nn.BatchNorm2d(outplane)
#         if inplane == 256:
#             self.ds = nn.AvgPool2d(2,2, padding=(1,0))
#         else:
#             self.ds = nn.AvgPool2d(2,2, padding=0)
#
#     def forward(self, x):
#         # x = self.bn1(self.PWconv(self.ds(x))) + self.bn(self.act_layer(self.conv(x)))
#         x = self.bn1(self.PWconv1(self.ds(x))) + self.bn(self.act_layer(self.PWconv(self.DWconv(x))))
#         # no pool
#         # x = self.bn(self.act_layer(self.conv(x)))
#         return x
#
# class feauture_Reduction(nn.Module):
#     def __init__(self, inplane, outplane):
#         super(feauture_Reduction, self).__init__()
#         # print(inplane, outplane)
#         # self.bn1 = nn.BatchNorm2d(inplane)
#         self.DWconv = nn.Conv2d(inplane, outplane,kernel_size=3, stride=2, padding=1,groups=inplane)
#         self.PWconv = nn.Conv2d(outplane, outplane,kernel_size=1, stride=1, padding=0)
#         self.PWconv1 = nn.Conv2d(inplane, outplane,kernel_size=1, stride=1, padding=0)
#         self.ds = nn.AvgPool2d(2, 2, padding=0)
#         # self.act_layer = nn.PReLU(outplane)
#         self.bn = nn.BatchNorm2d(outplane)
#         self.bn1 = nn.BatchNorm2d(outplane)
#
#     def forward(self, x):
#         # print(x.shape)
#         # print(self.DWconv(x).shape)
#         x = self.bn1(self.PWconv1(self.ds(x))) + self.bn(self.PWconv(self.DWconv(x)))
#         # no pool
#         # x = self.bn(self.act_layer(self.conv(x)))
#         return x


class UpperFaceAttention(nn.Module):
    def __init__(self, C2, C3, C4, block):
        super(UpperFaceAttention, self).__init__()
        self.C1 = self._make_layers(block, [C2, C3], [C3, C4], 2)
        self.C2 = self._make_layers(block, [C3], [C4], 1)
        self.Conv = nn.Sequential(
            nn.BatchNorm2d(C4 * 3),
            nn.Conv2d(C4 * 3, C4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(C4),
            nn.Dropout(0.3),
            nn.PReLU(C4),
            nn.Conv2d(C4, C4, kernel_size=1, stride=1),
            nn.BatchNorm2d(C4)
        )

        self.ms1 = Block_unit(dim=C4, num_heads=4, mlp_ratio=3, qkv_bias=False, qk_scale=None,
                              drop=.1, attn_drop=.1, drop_path=.0)
        self.ms2 = Block_unit(dim=C4, num_heads=4, mlp_ratio=3, qkv_bias=False, qk_scale=None,
                              drop=.1, attn_drop=.1, drop_path=.0)
        self.ms3 = Block_unit(dim=C4, num_heads=4, mlp_ratio=3, qkv_bias=False, qk_scale=None,
                              drop=.1, attn_drop=.1, drop_path=.0)

    def _make_layers(self, block, inplanes, outplanes, blocks):
        layers = []
        for i in range(blocks):
            # layers.append(block(inplanes[i], outplanes[i],ufa=True))
            layers.append(block(inplanes[i], outplanes[i]))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        x1, x2, x3 = inputs
        #feature-alignment
        x1 = self.C1(x1)
        x2 = self.C2(x2)
        B,C,H,W = x1.shape

        x1 = self.ms1(x1.reshape(B,C,H*W).permute(0,2,1)).reshape(B,C,H,W)
        x2 = self.ms2(x2.reshape(B,C,H*W).permute(0,2,1)).reshape(B,C,H,W)
        x3 = self.ms3(x3.reshape(B,C,H*W).permute(0,2,1)).reshape(B,C,H,W)
        # feature fusion
        ufa = torch.concat([x1,x2],dim=1)
        ufa = torch.concat([ufa,x3],dim=1)
        ufa = self.Conv(ufa)
        return ufa

class BlockIR(nn.Module):
    def __init__(self, inplanes, planes, stride, dim_match, use_att=False,
                 embed_dim=256, num_heads=4, mlp_ratio=4.,qkv_bias=True,
                 qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0, fcu=False):
        super(BlockIR, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False,groups=inplanes)
        self.conv1_1 = nn.Conv2d(planes,planes, kernel_size=1,stride=1,padding=0,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,groups=planes)
        self.conv2_1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.prelu2 = nn.PReLU(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.squeeze_block = F2S(inplanes=inplanes, outplanes=embed_dim, dw_stride=1)
        self.expand_block = S2F(inplanes=embed_dim, outplanes=planes, stride=stride)
        self.trans_block = Block_unit(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)
        if use_att:
            self.ecbam = Se(planes)
        else:
            self.ecbam = None
        self.fcu = fcu

        if dim_match:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        residual = x

        out = self.bn1(x)

        if self.fcu:
            _, _, H, W = out.shape
            x_st = self.squeeze_block(out)
            x_t = self.trans_block(x_st)
            x_t_r = self.expand_block(x_t, H, W)
        out = self.conv1(out)
        out = self.conv1_1(out)
        out = self.bn2(out)
        out = self.prelu1(out)
        # else:
        out = self.conv2(out)
        out = self.conv2_1(out)
        out = self.bn3(out)
        if self.fcu:
            out = self.prelu2(out)
            out = out + x_t_r
            out = self.conv3(out)
            out = self.bn4(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class LResNet(nn.Module):

    def __init__(self, block, layers, filter_list, is_gray=False):
        self.inplanes = 64
        super(LResNet, self).__init__()
        if is_gray:

            self.conv1 = nn.Conv2d(1, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)  # gray
        else:
            self.conv1 = nn.Conv2d(3, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        self.prelu1 = nn.PReLU(filter_list[0])
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2, fcu=False, use_att=False)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2, fcu=True, use_att=False)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2, fcu=True, use_att=False)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2, fcu=True, use_att=False)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 7 * 7),
            nn.Dropout(p=0.4),
            nn.Linear(filter_list[4] * 7 * 7, 512),
            nn.BatchNorm1d(512),  # fix gamma ???
        )
        self.fc_ufa = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 4 * 7),
            nn.Dropout(p=0.4),
            nn.Linear(filter_list[4] * 4 * 7, 512),
            nn.BatchNorm1d(512),  # fix gamma ???
        )
        self.ufa = UpperFaceAttention(filter_list[2], filter_list[3], filter_list[4], feauture_Reduction)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)


    def _make_layer(self, block, inplanes, planes, blocks, stride, fcu=False, use_att=False):
        layers = []
        layers.append(feauture_Reduction(inplanes, planes,ufa=False))
        for i in range(0, blocks):
            # print(i)
            layers.append(block(planes, planes, stride=1, dim_match=True, fcu=fcu, use_att=use_att))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.layer1(x)

        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x = self.layer4(x3)

        ufa = self.ufa([x2[:,:,:14,:], x3[:,:,:7,:], x[:,:,:4,:]])
        ufa = ufa.contiguous().view(ufa.size(0), -1)
        ufa = self.fc_ufa(ufa)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, ufa

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


def LResNet50_LDCSN(is_gray=False):
    filter_list = [64, 64, 128, 256, 512]
    layers = [2, 3, 13, 2]
    return LResNet(BlockIR, layers, filter_list, is_gray)
