from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class ModifiedGDC(nn.Module):
    def __init__(self, image_size, in_chs, num_classes, dropout, emb=512):  # embedding = 512 from original code
        super(ModifiedGDC, self).__init__()
        self.dropout = dropout

        # self.conv_dw = nn.Conv2d(in_chs, in_chs, kernel_size=1, groups=in_chs, bias=False)
        # self.bn1 = nn.BatchNorm2d(in_chs)
        # self.conv = nn.Conv2d(in_chs, emb, kernel_size=1, bias=False)

        if image_size % 32 == 0:
            flattened_features = emb * ((image_size // 32) ** 2)
        else:
            flattened_features = emb * ((image_size // 32 + 1) ** 2)

        self.bn2 = nn.BatchNorm1d(flattened_features)
        self.linear = nn.Linear(flattened_features, num_classes) if num_classes else nn.Identity()

    def forward(self, x):
        # x = self.conv_dw(x)
        # x = self.bn1(x)
        # x = self.conv(x)
        # x = x.view(x.size(0), -1)  # flatten
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn2(x)
        x = self.linear(x)
        return x

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.40, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)
        return output

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output
class ElasticArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.40,std=0.05):
        super(ElasticArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.std=std

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_theta.device) # Fast converge .clamp(self.m-self.std, self.m+self.std)
        m_hot.scatter_(1, label[index, None], margin)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta

def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1,w2).clamp(min=eps)


class CosMarginProduct_1(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.35,std=0.05):
        super(CosMarginProduct_1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.std = std
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(),
                              device=cosine.device)  # Fast converge .clamp(self.m-self.std, self.m+self.std)
        m_hot.scatter_(1, label[index, None], margin)
        phi = cosine - m_hot
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        # print(one_hot.shape, label.shape)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output[0])
        # output+=b
        # cosine = cosine_sim(input, self.weight)
        # # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # # --------------------------- convert label to one-hot ---------------------------
        # # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        # one_hot = torch.zeros_like(cosine)
        # index = torch.where(label!=-1)[0]
        # margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cosine.device)
        # one_hot.scatter_(1, label.view(-1, 1), margin)
        # # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # output = self.s * (cosine - one_hot)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'

class CosMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super(CosMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # print(self.weight.size(),input.size())
        # print(input)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class SphereFace2(nn.Module):
    """ reference: <SphereFace2: Binary Classification is All You Need
                    for Deep Face Recognition>
        margin='C' -> SphereFace2-C
        margin='A' -> SphereFace2-A
        marign='M' -> SphereFAce2-M
    """

    def __init__(self, feat_dim, num_class, magn_type='C',
                 alpha=0.7, r=40., m=0.4, t=3., lw=10.):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.magn_type = magn_type

        # alpha is the lambda in paper Eqn. 5
        self.alpha = alpha
        self.r = r
        self.m = m
        self.t = t
        self.lw = lw

        # init weights
        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

        # init bias
        z = alpha / ((1. - alpha) * (num_class - 1.))
        if magn_type == 'C':
            ay = r * (2. * 0.5 ** t - 1. - m)
            ai = r * (2. * 0.5 ** t - 1. + m)
        elif magn_type == 'A':
            theta_y = min(math.pi, math.pi / 2. + m)
            ay = r * (2. * ((math.cos(theta_y) + 1.) / 2.) ** t - 1.)
            ai = r * (2. * 0.5 ** t - 1.)
        elif magn_type == 'M':
            theta_y = min(math.pi, m * math.pi / 2.)
            ay = r * (2. * ((math.cos(theta_y) + 1.) / 2.) ** t - 1.)
            ai = r * (2. * 0.5 ** t - 1.)
        else:
            raise NotImplementedError

        temp = (1. - z) ** 2 + 4. * z * math.exp(ay - ai)
        b = (math.log(2. * z) - ai
             - math.log(1. - z + math.sqrt(temp)))
        self.b = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.b, b)

    def forward(self, x, y):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        # delta theta with margin
        cos_theta = F.normalize(x, dim=1).mm(self.w)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, y.view(-1, 1), 1.)
        with torch.no_grad():
            if self.magn_type == 'C':
                g_cos_theta = 2. * ((cos_theta + 1.) / 2.).pow(self.t) - 1.
                g_cos_theta = g_cos_theta - self.m * (2. * one_hot - 1.)
            elif self.magn_type == 'A':
                theta_m = torch.acos(cos_theta.clamp(-1 + 1e-5, 1. - 1e-5))
                theta_m.scatter_(1, y.view(-1, 1), self.m, reduce='add')
                theta_m.clamp_(1e-5, 3.14159)
                g_cos_theta = torch.cos(theta_m)
                g_cos_theta = 2. * ((g_cos_theta + 1.) / 2.).pow(self.t) - 1.
            elif self.magn_type == 'M':
                m_theta = torch.acos(cos_theta.clamp(-1 + 1e-5, 1. - 1e-5))
                m_theta.scatter_(1, y.view(-1, 1), self.m, reduce='multiply')
                m_theta.clamp_(1e-5, 3.14159)
                g_cos_theta = torch.cos(m_theta)
                g_cos_theta = 2. * ((g_cos_theta + 1.) / 2.).pow(self.t) - 1.
            else:
                raise NotImplementedError
            d_theta = g_cos_theta - cos_theta

        logits = self.r * (cos_theta + d_theta) + self.b
        weight = self.alpha * one_hot + (1. - self.alpha) * (1. - one_hot)
        weight = self.lw * self.num_class / self.r * weight
        loss = F.binary_cross_entropy_with_logits(
            logits, one_hot, weight=weight)

        return loss, logits

class SphereMarginProduct(nn.Module):
    r"""Implement of SphereFace loss :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin [1.5, 4.0]
        cos(m * theta)
    """

    def __init__(self, in_features, out_features, s=64, m=1.5):
        super(SphereMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = cosine_sim(input, self.weight)
        # --------------------------- convert label to one-hot ---------------------------
        theta = self.m * torch.acos(cosine)
        cosine_new = torch.cos(theta)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * cosine_new) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

class criterion_mask(nn.Module):
    """
    in order to make two masks be different
    """
    def __init__(self, IoU_thres):
        super(criterion_mask, self).__init__()
        self.IoU_thres = IoU_thres
        self.criterion_diff = torch.nn.L1Loss(reduction='none')

    def forward(self, mask1, mask2, IoU):
        IoU = IoU.cuda()
        loss_zeros = torch.zeros(mask1.size(0)).double().cuda()
        # loss_small = torch.sum((1-mask1) * (1-mask2), [1, 2, 3], dtype=float) / mask1[0].nelement()
        # loss_small = torch.sum((1-mask1) * (1-mask2), [1, 2, 3], dtype=float) / (torch.sum((1-mask1)+(1-mask2), dtype=float) + 1e-5)
        A = torch.sum((1-mask1)*(1-mask2), [1, 2, 3], dtype=float)
        B = torch.sum((1-mask1)+(1-mask2), [1, 2, 3], dtype=float) + 1e-5
        loss_small = A / torch.clamp(B, min=0.0, max=1.0)
        loss_small = loss_small.double().cuda()

        # loss_small = self.criterion_diff(mask1, (1-mask2))
        # loss_small = torch.sum(loss_small, [1, 2, 3]).double() / loss_small[0].nelement()

        loss_big = self.criterion_diff(mask1, mask2)
        loss_big = torch.sum(loss_big, [1, 2, 3]).double() / loss_big[0].nelement()

        loss_small = torch.where(IoU < self.IoU_thres[0], loss_small, loss_zeros)
        loss_big = torch.where(IoU > self.IoU_thres[1], loss_big, loss_zeros)

        loss = loss_small + loss_big

        return torch.mean(loss)

