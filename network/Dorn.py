# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 12:33
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math

from torch.nn import BatchNorm2d
from pathlib import Path
from torch.nn.modules.utils import _pair, _triple
import numpy as np


def consistent_padding_with_dilation(padding, dilation, dim=2):
    assert dim == 2 or dim == 3, 'Convolution layer only support 2D and 3D'
    if dim == 2:
        padding = _pair(padding)
        dilation = _pair(dilation)
    else:  # dim == 3
        padding = _triple(padding)
        dilation = _triple(dilation)

    padding = list(padding)
    for d in range(dim):
        padding[d] = dilation[d] if dilation[d] > 1 else padding[d]
    padding = tuple(padding)

    return padding, dilation


def conv_bn_relu(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    padding, dilation = consistent_padding_with_dilation(padding, dilation, dim=2)
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias),
            nn.ReLU(inplace=True),
        )


class FullImageEncoder(nn.Module):
    def __init__(self, h, w, kernel_size, dropout_prob=0.5):
        super(FullImageEncoder, self).__init__()
        self.global_pooling = nn.AvgPool2d(kernel_size, stride=kernel_size, padding=kernel_size // 2)  # KITTI 16 16
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.h = h // kernel_size + 1
        self.w = w // kernel_size + 1
        # print("h=", self.h, " w=", self.w, h, w)
        self.global_fc = nn.Linear(2048 * self.h * self.w, 512)  # kitti 4x5
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)  # 1x1 卷积

    def forward(self, x):
        # print('x size:', x.size())
        x1 = self.global_pooling(x)
        # print('# x1 size:', x1.size())
        x2 = self.dropout(x1)
        x3 = x2.view(-1, 2048 * self.h * self.w)  # kitti 4x5
        x4 = self.relu(self.global_fc(x3))
        # print('# x4 size:', x4.size())
        x4 = x4.view(-1, 512, 1, 1)
        # print('# x4 size:', x4.size())
        x5 = self.conv1(x4)
        # out = self.upsample(x5)
        return x5


class SceneUnderstandingModule(nn.Module):
    def __init__(self, ord_num, size, kernel_size, pyramid=[6, 12, 18], dropout_prob=0.5, batch_norm=False):
        # pyramid kitti [6, 12, 18] nyu [4, 8, 12]
        super(SceneUnderstandingModule, self).__init__()
        assert len(size) == 2
        assert len(pyramid) == 3
        self.size = size
        h, w = self.size
        self.encoder = FullImageEncoder(h // 8, w // 8, kernel_size, dropout_prob)
        self.aspp1 = nn.Sequential(
            conv_bn_relu(batch_norm, 2048, 512, kernel_size=1, padding=0),
            conv_bn_relu(batch_norm, 512, 512, kernel_size=1, padding=0)
        )
        self.aspp2 = nn.Sequential(
            conv_bn_relu(batch_norm, 2048, 512, kernel_size=3, padding=pyramid[0], dilation=pyramid[0]),
            conv_bn_relu(batch_norm, 512, 512, kernel_size=1, padding=0)
        )
        self.aspp3 = nn.Sequential(
            conv_bn_relu(batch_norm, 2048, 512, kernel_size=3, padding=pyramid[1], dilation=pyramid[1]),
            conv_bn_relu(batch_norm, 512, 512, kernel_size=1, padding=0)
        )
        self.aspp4 = nn.Sequential(
            conv_bn_relu(batch_norm, 2048, 512, kernel_size=3, padding=pyramid[2], dilation=pyramid[2]),
            conv_bn_relu(batch_norm, 512, 512, kernel_size=1, padding=0)
        )
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=dropout_prob),
            conv_bn_relu(batch_norm, 512 * 5, 2048, kernel_size=1, padding=0),
            nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(2048, int(ord_num * 2), 1)
        )

    def forward(self, x):
        N, C, H, W = x.shape
        x1 = self.encoder(x)
        x1 = F.interpolate(x1, size=(H, W), mode="bilinear", align_corners=True)

        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)

        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.concat_process(x6)
        out = F.interpolate(out, size=self.size, mode="bilinear", align_corners=True)
        return out

affine_par = True


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out

def resnet101(pretrained=True):
    resnet101 = ResNet(Bottleneck, [3, 4, 23, 3])

    if pretrained:
        weights_file = Path('./network/pretrained_models/resnet101-imagenet.pth').resolve()
        if not weights_file.exists():
            import urllib
            weights_file.parent.mkdir(parents=True)
            urllib.request.urlretrieve("http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth", weights_file.as_posix())
        saved_state_dict = torch.load(weights_file.as_posix())
        # saved_state_dict = torch.load('./pretrained_models/resnet101-imagenet.pth')
        new_params = resnet101.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[0] == 'fc':
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

        resnet101.load_state_dict(new_params)

    return resnet101

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = ResNet(Bottleneck, [3, 4, 23, 3])

        if pretrained:
            weights_file = Path('./network/pretrained_models/resnet101-imagenet.pth').resolve()
            if not weights_file.exists():
                import urllib
                weights_file.parent.mkdir(parents=True)
                urllib.request.urlretrieve("http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth", weights_file.as_posix())
            saved_state_dict = torch.load(weights_file.as_posix())
            # saved_state_dict = torch.load('./pretrained_models/resnet101-imagenet.pth')
            new_params = self.backbone.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[0] == 'fc':
                    new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

            self.backbone.load_state_dict(new_params)

    def forward(self, input):
        return self.backbone(input)

class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def freeze(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        """
        :param x: NxCxHxW, N is batch_size, C is channels of features
        :return: ord_label is ordinal outputs for each spatial locations , N x 1 x H x W
                 ord prob is the probability of each label, N x OrdNum x H x W
        """
        N, C, H, W = x.size()
        ord_num = C // 2

        # implementation according to the paper
        # A = x[:, ::2, :, :]
        # B = x[:, 1::2, :, :]
        #
        # # A = A.reshape(N, 1, ord_num * H * W)
        # # B = B.reshape(N, 1, ord_num * H * W)
        # A = A.unsqueeze(dim=1)
        # B = B.unsqueeze(dim=1)
        # concat_feats = torch.cat((A, B), dim=1)
        #
        # if self.training:
        #     prob = F.log_softmax(concat_feats, dim=1)
        #     ord_prob = x.clone()
        #     ord_prob[:, 0::2, :, :] = prob[:, 0, :, :, :]
        #     ord_prob[:, 1::2, :, :] = prob[:, 1, :, :, :]
        #     return ord_prob
        #
        # ord_prob = F.softmax(concat_feats, dim=1)[:, 0, ::]
        # ord_label = torch.sum((ord_prob > 0.5), dim=1).reshape((N, 1, H, W))
        # return ord_prob, ord_label

        # reimplementation for fast speed.

        x = x.view(-1, 2, ord_num, H, W)
        prob = F.log_softmax(x, dim=1).view(N, C, H, W)

        ord_prob = F.softmax(x, dim=1)[:, 0, :, :, :]
        ord_label = torch.sum((ord_prob > 0.5), dim=1)
        return prob, ord_prob, ord_label


class DORN(nn.Module):

    def __init__(self, args):
        self.args = args
        super().__init__()
        assert len(self.args.input_size) == 2
        assert isinstance(self.args.kernel_size, int)
        self.ord_num = self.args.ord_num
        self.gamma = self.args.gamma
        self.beta = self.args.beta
        self.discretization = self.args.discretization

        self.backbone = ResNetBackbone(pretrained=self.args.pretrained)
        self.SceneUnderstandingModule = SceneUnderstandingModule(self.ord_num, size=self.args.input_size,
                                                                 kernel_size=self.args.kernel_size,
                                                                 pyramid=self.args.pyramid,
                                                                 batch_norm=self.args.batch_norm)
        self.regression_layer = OrdinalRegressionLayer()

    def forward(self, image):
        """
        :param image: RGB image, torch.Tensor, Nx3xHxW
        :param target: ground truth depth, torch.Tensor, NxHxW
        :return: output: if training, return loss, torch.Float,
                         else return {"target": depth, "prob": prob, "label": label},
                         depth: predicted depth, torch.Tensor, NxHxW
                         prob: probability of each label, torch.Tensor, NxCxHxW, C is number of label
                         label: predicted label, torch.Tensor, NxHxW
        """
        N, C, H, W = image.shape
        feat = self.backbone(image)
        feat = self.SceneUnderstandingModule(feat)
        # print("feat shape:", feat.shape)
        # feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=True)

        prob, ord_prob, label = self.regression_layer(feat)
        # print("prob shape:", prob.shape, " label shape:", label.shape)
        if self.discretization == "SID":
            t0 = torch.exp(np.log(self.beta) * label.float() / self.ord_num)
            t1 = torch.exp(np.log(self.beta) * (label.float() + 1) / self.ord_num)
        else:
            t0 = 1.0 + (self.beta - 1.0) * label.float() / self.ord_num
            t1 = 1.0 + (self.beta - 1.0) * (label.float() + 1) / self.ord_num
        depth = (t0 + t1) / 2 - self.gamma
        depth = depth.unsqueeze(1)
        # print("depth min:", torch.min(depth), " max:", torch.max(depth),
        #       " label min:", torch.min(label), " max:", torch.max(label))
        return prob, depth


if __name__ == "__main__":
    dorn = DORN().cuda()
    img = torch.rand((1, 3, 64, 64)).cuda()
    dorn(img)