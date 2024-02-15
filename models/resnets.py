import torch
import torch.nn as nn

class BlockIR(nn.Module):
    def __init__(self, inplanes, planes, stride, dim_match):
        super(BlockIR, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

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
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class LResNet_Occ(nn.Module):

    def __init__(self, block, layers, filter_list, is_gray=False):
        self.inplanes = 64
        super(LResNet_Occ, self).__init__()
        # input is (mini-batch,3 or 1,112,96)
        # use (conv3x3, stride=1, padding=1) instead of (conv7x7, stride=2, padding=3)
        if is_gray:
            self.conv1 = nn.Conv2d(1, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)  # gray
        else:
            self.conv1 = nn.Conv2d(3, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        self.prelu1 = nn.PReLU(filter_list[0])
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)
        # --Begin--Triplet branch 
        # self.triplet = nn.Sequential(
        #     #nn.BatchNorm2d(filter_list[4]),
        #     nn.Conv2d(filter_list[4], 512, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PReLU(filter_list[4]),
        #     nn.BatchNorm2d(filter_list[4]),
        #     nn.Sigmoid(),
        # )
        # self.regress = nn.Sequential(
            # nn.BatchNorm1d(filter_list[4]*7*6),
            # nn.Linear(filter_list[4]*7*6, 512, bias=False),
            # nn.BatchNorm1d(512),
            # nn.Linear(512, filter_list[5], bias=False),
            # nn.BatchNorm1d(filter_list[5]),
        # )
        # self.regress = nn.Sequential(
        #     nn.BatchNorm1d(filter_list[4]*7*6),
        #     nn.Dropout(p=0.5),  # No drop for triplet dic
        #     nn.Linear(filter_list[4]*7*6, filter_list[5], bias=False),
        #     nn.BatchNorm1d(filter_list[5]),
        # )
        # --End--Triplet branch
        self.fc = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 8 * 8),
            nn.Dropout(p=0.5),  # No drop for triplet dic
            nn.Linear(filter_list[4] * 8 * 8, 512),
            nn.BatchNorm1d(512),  # fix gamma ???
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)


    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(block(inplanes, planes, stride, False))
        for i in range(1, blocks):
            layers.append(block(planes, planes, stride=1, dim_match=True))

        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        fmap = self.layer4(x)

        # generate mask
        # if not isinstance(mask, torch.Tensor):
        #     mask = self.triplet(fmap)

        # regress
        # vec = self.regress(mask.view(mask.size(0), -1))
        #
        # fmap_mask = fmap * mask
        #
        # fc_mask = self.fc(fmap_mask.view(fmap_mask.size(0), -1))

        fc = self.fc(fmap.view(fmap.size(0), -1))

        # return fc_mask, mask, vec, fc
        return fc

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


def LResNet50E_IR_Occ(is_gray=False, num_mask=101):
    filter_list = [64, 64, 128, 256, 512, num_mask]
    layers = [3, 4, 14, 3]
    model = LResNet_Occ(BlockIR, layers, filter_list, is_gray)
    return model 

# -------------------------------------- LResNet50E-IR network Begin ----------------------------------------
class LResNet(nn.Module):

    def __init__(self, block, layers, filter_list, is_gray=False):
        self.inplanes = 64
        super(LResNet, self).__init__()
        # input is (mini-batch,3 or 1,112,96)
        # use (conv3x3, stride=1, padding=1) instead of (conv7x7, stride=2, padding=3)
        if is_gray:
            self.conv1 = nn.Conv2d(1, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)  # gray
        else:
            self.conv1 = nn.Conv2d(3, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        self.prelu1 = nn.PReLU(filter_list[0])
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 7 * 6),
            nn.Dropout(p=0.4),
            nn.Linear(filter_list[4] * 7 * 6, 512),
            nn.BatchNorm1d(512),  # fix gamma ???
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)


    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(block(inplanes, planes, stride, False))
        for i in range(1, blocks):
            layers.append(block(planes, planes, stride=1, dim_match=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


def LResNet50E_IR(is_gray=False):
    filter_list = [64, 64, 128, 256, 512]
    layers = [3, 4, 14, 3]
    return LResNet(BlockIR, layers, filter_list, is_gray)
# ---------------------------------- LResNet50E-IR network End ----------------------------------

class LResNet_Occ_Song(nn.Module):

    def __init__(self, block, layers, filter_list, is_gray=False):
        self.inplanes = 64
        super(LResNet_Occ_Song, self).__init__()
        # input is (mini-batch,3 or 1,112,96)
        # use (conv3x3, stride=1, padding=1) instead of (conv7x7, stride=2, padding=3)
        if is_gray:
            self.conv1 = nn.Conv2d(1, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)  # gray
        else:
            self.conv1 = nn.Conv2d(3, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        self.prelu1 = nn.PReLU(filter_list[0])
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)
        # --Begin--Triplet branch 
        self.triplet = nn.Sequential(
            #nn.BatchNorm2d(filter_list[4]),
            nn.Conv2d(filter_list[4], 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(filter_list[4]),
            nn.BatchNorm2d(filter_list[4]),
            nn.Sigmoid(),
        ) 
        # self.regress = nn.Sequential(
            # nn.BatchNorm1d(filter_list[4]*7*6),
            # nn.Linear(filter_list[4]*7*6, 512, bias=False),
            # nn.BatchNorm1d(512),
            # nn.Linear(512, filter_list[5], bias=False),
            # nn.BatchNorm1d(filter_list[5]),
        # )
        # self.regress = nn.Sequential(
            # nn.BatchNorm1d(filter_list[4]*7*6),
            # nn.Dropout(p=0.5),  # No drop for triplet dic
            # nn.Linear(filter_list[4]*7*6, filter_list[5], bias=False),
            # nn.BatchNorm1d(filter_list[5]),
        # )
        # --End--Triplet branch
        self.fc = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 7 * 6),
            nn.Dropout(p=0.5),  # No drop for triplet dic
            nn.Linear(filter_list[4] * 7 * 6, 512),
            nn.BatchNorm1d(512),  # fix gamma ???
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)


    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(block(inplanes, planes, stride, False))
        for i in range(1, blocks):
            layers.append(block(planes, planes, stride=1, dim_match=True))

        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        fmap = self.layer4(x)

        # generate mask
        if not isinstance(mask, torch.Tensor):
            mask = self.triplet(fmap)

        # regress
        vec = 0

        fmap_mask = fmap * mask

        fc_mask = self.fc(fmap_mask.view(fmap_mask.size(0), -1))

        fc = self.fc(fmap.view(fmap.size(0), -1))

        return fc_mask, mask, vec, fc 

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


def LResNet50E_IR_Occ_Song(is_gray=False, num_mask=101):
    filter_list = [64, 64, 128, 256, 512, num_mask]
    layers = [3, 4, 14, 3]
    model = LResNet_Occ_Song(BlockIR, layers, filter_list, is_gray)
    return model 
