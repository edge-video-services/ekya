import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.utils.checkpoint import *
import torch.nn.functional as F
from torch.nn import init
import numpy as np


class ASPP(nn.Module):
    def __init__(self, C, depth, num_classes, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        super(ASPP, self).__init__()
        self._C = C
        self._depth = depth
        self._num_classes = num_classes
        self._norm = norm

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = nn.Conv2d(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp2 = nn.Conv2d(C, depth, kernel_size=3, stride=1,
                               dilation=int(6*mult), padding=int(6*mult),
                               bias=False)
        self.aspp3 = nn.Conv2d(C, depth, kernel_size=3, stride=1,
                               dilation=int(12*mult), padding=int(12*mult),
                               bias=False)
        self.aspp4 = nn.Conv2d(C, depth, kernel_size=3, stride=1,
                               dilation=int(18*mult), padding=int(18*mult),
                               bias=False)
        self.aspp5 = nn.Conv2d(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = self._norm(depth, momentum)
        self.aspp2_bn = self._norm(depth, momentum)
        self.aspp3_bn = self._norm(depth, momentum)
        self.aspp4_bn = self._norm(depth, momentum)
        self.aspp5_bn = self._norm(depth, momentum)
        self.conv2 = nn.Conv2d(depth * 5, depth, kernel_size=1, stride=1,
                               bias=False)
        self.bn2 = self._norm(depth, momentum)
        self.conv3 = nn.Conv2d(depth, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',
                         align_corners=True)(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x


class Selayer(nn.Module):

    def __init__(self, inplanes):
        super(Selayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.global_avgpool(x)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out


class BottleneckX(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None, dilation=1, norm=None, elastic=False, se=False):
        super(BottleneckX, self).__init__()
        self.se = se
        self.elastic = elastic and stride == 1 and planes < 512
        if self.elastic:
            self.down = nn.AvgPool2d(2, stride=2)
            self.ups = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # half resolution
        self.conv1_d = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1_d = norm(planes)
        self.conv2_d = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, groups=cardinality // 2,
                                 dilation=dilation, padding=dilation, bias=False)
        self.bn2_d = norm(planes)
        self.conv3_d = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        # full resolution
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, groups=cardinality // 2,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        # after merging
        self.bn3 = norm(planes * self.expansion)
        if self.se:
            self.selayer = Selayer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.__flops__ = 0

    def forward(self, x):
        residual = x
        out_d = x
        if self.elastic:
            if x.size(2) % 2 > 0 or x.size(3) % 2 > 0:
                out_d = F.pad(out_d, (0, x.size(3) % 2, 0, x.size(2) % 2), mode='replicate')
            out_d = self.down(out_d)

        out_d = self.conv1_d(out_d)
        out_d = self.bn1_d(out_d)
        out_d = self.relu(out_d)

        out_d = self.conv2_d(out_d)
        out_d = self.bn2_d(out_d)
        out_d = self.relu(out_d)

        out_d = self.conv3_d(out_d)

        if self.elastic:
            out_d = self.ups(out_d)
            self.__flops__ += np.prod(out_d[0].shape) * 8
            if out_d.size(2) > x.size(2) or out_d.size(3) > x.size(3):
                out_d = out_d[:, :, :x.size(2), :x.size(3)]

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = out + out_d
        out = self.bn3(out)

        if self.se:
            out = self.selayer(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNext(nn.Module):

    def __init__(self, block, layers, num_classes=1000, seg=False, elastic=False, se=False):
        self.inplanes = 64
        self.cardinality = 32
        self.seg = seg
        self._norm = lambda planes, momentum=0.05 if seg else 0.1: torch.nn.BatchNorm2d(planes, momentum=momentum)

        super(ResNext, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], elastic=elastic, se=se)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, elastic=elastic, se=se)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, elastic=elastic, se=se)
        if seg:
            self.layer4 = self._make_mg(block, 512, se=se)
            self.aspp = ASPP(512 * block.expansion, 256, num_classes, self._norm)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, torch.nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        else:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, elastic=False, se=se)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            init.normal_(self.fc.weight, std=0.01)
            for n, p in self.named_parameters():
                if n.split('.')[-1] == 'weight':
                    if 'conv' in n:
                        init.kaiming_normal_(p, mode='fan_in', nonlinearity='relu')
                    if 'bn' in n:
                        p.data.fill_(1)
                    if 'bn3' in n:
                        p.data.fill_(0)
                elif n.split('.')[-1] == 'bias':
                    p.data.fill_(0)

    def _make_layer(self, block, planes, blocks, stride=1, elastic=False, se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self._norm(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, self.cardinality, stride, downsample=downsample, norm=self._norm, elastic=elastic, se=se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, norm=self._norm, elastic=elastic, se=se))
        return nn.Sequential(*layers)

    def _make_mg(self, block, planes, dilation=2, multi_grid=(1, 2, 4), se=False):
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=1, dilation=1, bias=False),
            self._norm(planes * block.expansion),
        )

        layers = list()
        layers.append(block(self.inplanes, planes, self.cardinality, downsample=downsample, dilation=dilation*multi_grid[0], norm=self._norm, se=se))
        self.inplanes = planes * block.expansion
        layers.append(block(self.inplanes, planes, self.cardinality, dilation=dilation*multi_grid[1], norm=self._norm, se=se))
        layers.append(block(self.inplanes, planes, self.cardinality, dilation=dilation*multi_grid[2], norm=self._norm, se=se))
        return nn.Sequential(*layers)

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.seg:
            for module in self.layer1._modules.values():
                x = checkpoint(module, x)
            for module in self.layer2._modules.values():
                x = checkpoint(module, x)
            for module in self.layer3._modules.values():
                x = checkpoint(module, x)
            for module in self.layer4._modules.values():
                x = checkpoint(module, x)
            x = self.aspp(x)
            x = nn.Upsample(size, mode='bilinear', align_corners=True)(x)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


def resnext50(seg=False, **kwargs):
    model = ResNext(BottleneckX, [3, 4, 6, 3], seg=seg, elastic=False, **kwargs)
    return model


def se_resnext50(seg=False, **kwargs):
    model = ResNext(BottleneckX, [3, 4, 6, 3], seg=seg, elastic=False, se=True, **kwargs)
    return model


def resnext50_elastic(seg=False, **kwargs):
    model = ResNext(BottleneckX, [6, 8, 5, 3], seg=seg, elastic=True, **kwargs)
    return model


def se_resnext50_elastic(seg=False, **kwargs):
    model = ResNext(BottleneckX, [6, 8, 5, 3], seg=seg, elastic=True, se=True, **kwargs)
    return model


def resnext101(seg=False, **kwargs):
    model = ResNext(BottleneckX, [3, 4, 23, 3], seg=seg, elastic=False, **kwargs)
    return model


def resnext101_elastic(seg=False, **kwargs):
    model = ResNext(BottleneckX, [12, 14, 20, 3], seg=seg, elastic=True, **kwargs)
    return model
