import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_conv(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, stride=1, padding=0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # self.proj1 = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, C * 3, N).permute(0, 2, 1)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_CV(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (v @ attn).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_unit(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # self.proj1 = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.proj_drop1 = nn.Dropout(proj_drop)
        #fc_q_k
        # self.q_proj = nn.Linear(dim, dim)
        # self.k_proj = nn.Linear(dim, dim)
        # self.v_proj = nn.Linear(dim, dim)


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # deep unit space-channel
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        v1 = (attn @ v)#.transpose(1, 2).reshape(B, N, C)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (v1 @ attn).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # deep unit channel-space
        # attn = (q.transpose(-2, -1) @ k) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        #
        # v1 = (v @ attn)  # .transpose(1, 2).reshape(B, N, C)
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = (attn @ v1).transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)

        # deep unit
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        #
        # x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x_attn = self.proj(x_attn)
        # x_attn = self.proj_drop(x_attn)
        # x = x + x_attn
        #
        # #fc_q
        # q = self.q_proj(q.permute(0, 2, 1, 3).reshape(B, N, C)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # k = self.k_proj(k.permute(0, 2, 1, 3).reshape(B, N, C)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # v = self.v_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # attn = (q.transpose(-2, -1) @ k) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        #
        # x_attn = (v @ attn).transpose(1, 2).reshape(B, N, C)
        # x_attn = self.proj1(x_attn)
        # x_attn = self.proj_drop1(x_attn)
        # x = x + x_attn

        return x

class Attention_unit_conv(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(dim*3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # self.proj1 = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.proj_drop1 = nn.Dropout(proj_drop)
        #fc_q_k
        # self.q_proj = nn.Linear(dim, dim)
        # self.k_proj = nn.Linear(dim, dim)
        # self.v_proj = nn.Linear(dim, dim)


    def forward(self, x):
        B, C, H, W = x.shape
        N = H*W
        qkv = self.qkv(x)
        qkv = self.bn(qkv).reshape(B,C*3,N).permute(0,2,1)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # deep unit
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        v1 = (attn @ v)#.transpose(1, 2).reshape(B, N, C)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (v1 @ attn).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # deep unit
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        #
        # x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x_attn = self.proj(x_attn)
        # x_attn = self.proj_drop(x_attn)
        # x = x + x_attn
        #
        # #fc_q
        # q = self.q_proj(q.permute(0, 2, 1, 3).reshape(B, N, C)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # k = self.k_proj(k.permute(0, 2, 1, 3).reshape(B, N, C)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # v = self.v_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # attn = (q.transpose(-2, -1) @ k) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        #
        # x_attn = (v @ attn).transpose(1, 2).reshape(B, N, C)
        # x_attn = self.proj1(x_attn)
        # x_attn = self.proj_drop1(x_attn)
        # x = x + x_attn

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.conv = torch.nn.Conv1d(dim,dim,kernel_size=5,stride=1,padding=2)
        # self.act_layer = act_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = self.act_layer(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        #deep norm
        # x = self.norm1(x + self.drop_path(self.attn(x)))
        # x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x

class Block_Conv(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_cv = Attention_CV(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # kernel_size = int(abs((math.log(dim,2)+1)/2))
        # if kernel_size % 2:
        #     kernel_size=kernel_size
        # else:
        #     kernel_size+=1


    def forward(self, x):
        #EMHSA_Conv
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.attn_cv(self.norm2(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        #EMHSA_1
        # x = x + self.drop_path(self.attn_cv(self.norm2(x)))
        # x = x + self.drop_path(self.attn(self.norm1(x)))

        return x

class Block_unit(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        #MMHSA
        self.attn = Attention_unit(
            dim, num_heads=num_heads,  attn_drop=attn_drop, proj_drop=drop)
        #MHSA
        # self.attn = Attention(dim, num_heads=num_heads,  attn_drop=attn_drop, proj_drop=drop)

        # self.attn_cv = multi_conv_self_attention_C(dim, heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)



    def forward(self, x):
        #Block_multi_Conv
        # print(x.shape)
        x = x + self.drop_path(self.attn(self.norm(x)))
        # x = self.drop_path(self.attn(self.norm(x)))


        return x

class Block_unit_MHSA(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        #MMHSA
        # self.attn = Attention_unit(
            # dim, num_heads=num_heads,  attn_drop=attn_drop, proj_drop=drop)
        #MHSA
        self.attn = Attention(dim, num_heads=num_heads,  attn_drop=attn_drop, proj_drop=drop)

        # self.attn_cv = multi_conv_self_attention_C(dim, heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)



    def forward(self, x):
        #Block_multi_Conv
        # print(x.shape)
        x = x + self.drop_path(self.attn(self.norm(x)))
        # x = self.drop_path(self.attn(self.norm(x)))


        return x

class Block_unit_conv(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.BatchNorm2d, eps=1e-6)):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        #CMMHSA
        self.attn = Attention_unit_conv(
            dim, num_heads=num_heads,  attn_drop=attn_drop, proj_drop=drop)
        #CMHSA
        # self.attn = Attention_conv(
        #     dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)

        # self.attn_cv = multi_conv_self_attention_C(dim, heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = nn.BatchNorm2d(dim)



    def forward(self, x):
        #Block_multi_Conv
        B,C,H,W = x.shape
        # print(x.shape)
        # print(self.drop_path(self.attn(self.norm(x))).shape)
        x = x.reshape(B, C, H*W).permute(0,2,1) + self.drop_path(self.attn(self.norm(x)))
        # x = self.drop_path(self.attn(self.norm(x)))


        return x

class FCU_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class WRLDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(WRLDown, self).__init__()
        # self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=2, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]
        # print(x.shape)
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        # print(x.shape)
        x = self.ln(x)
        x = self.act(x)
        # print(x.shape)
        # print(x.shape)
        # print(x_t.size(), x_t[:, 0][:, None, :].size(),x.size())
        # x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)
        # print(x.size())
        x =x + x_t


        return x

class F2S(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(F2S, self).__init__()
        self.dw_stride = dw_stride
        # print()
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=self.dw_stride, padding=0)
        # self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x):
        # print(x.shape)
        x = self.conv_project(x).flatten(2).transpose(1, 2)  # [N, C, H, W]- [N, H*W, C]

        # x = self.sample_pooling(x)
        x = self.ln(x)
        x = self.act(x)
        # print(x.shape)
        # print(x.shape)
        # print(x_t.size(), x_t[:, 0][:, None, :].size(),x.size())
        # x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)


        return x

class C2M(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.PReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6)):
        super(C2M, self).__init__()
        self.dw_stride = dw_stride
        # print()
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=self.dw_stride, padding=0)
        # self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.bn = nn.BatchNorm2d(outplanes)
        self.act = act_layer()

    def forward(self, x):
        # print(x.shape)
        x = self.conv_project(x)#.flatten(2).transpose(1, 2)  # [N, C, H, W]- [N, H*W, C]

        # x = self.sample_pooling(x)
        x = self.bn(x)
        x = self.act(x)
        # print(x.shape)
        # print(x.shape)
        # print(x_t.size(), x_t[:, 0][:, None, :].size(),x.size())
        # x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)


        return x

class WRLUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), ):
        super(WRLUp, self).__init__()

        # self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x.transpose(1, 2).reshape(B, C, H, W)
        # print(x_r.shape)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return x_r

class S2F(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), ):
        super(S2F, self).__init__()

        self.stride = stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=self.stride, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # print(x.shape)
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x.transpose(1, 2).reshape(B, C, H, W)
        # print(x_r.shape)
        x_r = self.act(self.bn(self.conv_project(x_r)))
        # print(x_r.shape)

        # return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))
        return x_r

class M2C(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, stride, act_layer=nn.PReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), ):
        super(M2C, self).__init__()

        self.stride = stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=self.stride, padding=0)
        self.bn = nn.BatchNorm2d(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # print(x.shape)
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x.transpose(1, 2).reshape(B, C, H, W)
        # print(x_r.shape)
        x_r = self.act(self.bn(self.conv_project(x_r)))
        # print(x_r.shape)

        # return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))
        return x_r

