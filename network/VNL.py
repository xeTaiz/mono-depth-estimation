import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
import math
from torch.nn import functional as F
from pathlib import Path
import numpy as np

def get_func(args):
    if args.encoder == 'resnext50_32x4d_body_stride16':
        return lateral(ResNeXt50_32x4d_body_stride16, args)
    elif args.encoder == 'resnext101_32x4d_body_stride16':
        return lateral(ResNeXt101_32x4d_body_stride16, args)
    elif args.encoder == 'mobilenetv2_body_stride8':
        return lateral(MobileNetV2_body_stride8, args)
    else:
        raise ValueError("Unknown bottom up model")


def convert_state_dict_mobilenet(src_dict):
        """Return the correct mapping of tensor name and value

        Mapping from the names of torchvision model to our resnet conv_body and box_head.
        """
        dst_dict = {}
        res_block_n = np.array([1, 4, 7, 14, 18])
        for k, v in src_dict.items():
            toks = k.split('.')
            id_n = int(toks[1])
            if id_n < 18 and '17.conv.7' not in k and 'classifier' not in k:
                res_n = np.where(res_block_n > id_n)[0][0] + 1
                n = res_n - 2 if res_n >= 2 else 0
                res_n_m = 0 if id_n - res_block_n[n] < 0 else id_n - res_block_n[n]
                toks[0] = 'res%s' % res_n
                toks[1] = '%s' % res_n_m
                name = '.'.join(toks)
                dst_dict[name] = v
        return dst_dict

def convert_state_dict_resnext(src_dict):
        """Return the correct mapping of tensor name and value

        Mapping from the names of torchvision model to our resnet conv_body and box_head.
        """
        dst_dict = {}
        res_id = 1
        map1 = ['conv1.', 'bn1.', ' ', 'conv2.', 'bn2.']
        map2 = [[' ', 'conv3.', 'bn3.'], ['shortcut.conv.', 'shortcut.bn.']]
        for k, v in src_dict.items():
            toks = k.split('.')
            if int(toks[0]) == 0:
                name = 'res%d.' % res_id + 'conv1.' + toks[-1]
            elif int(toks[0]) == 1:
                name = 'res%d.' % res_id + 'bn1.' + toks[-1]
            elif int(toks[0]) >=4 and int(toks[0]) <= 7:
                name_res = 'res%d.%d.' % (int(toks[0])-2, int(toks[1]))
                if len(toks) == 7:
                    name = name_res + map1[int(toks[-2])] + toks[-1]
                elif len(toks) == 6:
                    name = name_res + map2[int(toks[-3])][int(toks[-2])] + toks[-1]
            else:
                continue
            dst_dict[name] = v

        return dst_dict


def load_pretrained_imagenet_weights(model, encoder):
    net = "MobileNetV2-ImageNet" if "mobilenet" in encoder else "ResNeXt-ImageNet"
    if "resnext50" in encoder:
        weights_file = "resnext50_32x4d.pth"
    elif "resnext101" in encoder:
        weights_file = "resnext101_32x4d.pth"
    elif "mobilenet" in encoder:
        weights_file = "mobilenet_v2.pth.tar"
    else:
        raise ValueError("unknow encoder", encoder)
    weights_file = Path(Path.cwd(), "network", "pretrained_models", net, weights_file)
    convert_state_dict = convert_state_dict_mobilenet if "mobilenet" in encoder else convert_state_dict_resnext
    pretrianed_state_dict = convert_state_dict(torch.load(weights_file))

    model_state_dict = model.state_dict()

    for k, v in pretrianed_state_dict.items():
        if k in model_state_dict.keys():
            model_state_dict[k].copy_(pretrianed_state_dict[k])
        else:
            print('Weight %s is not in ResNeXt model.' % k)
    print('Pretrained {} weight has been loaded'.format(encoder))

class lateral(nn.Module):
    def __init__(self, conv_body_func, args):
        super().__init__()

        self.dim_in = args.enc_dim_in
        self.dim_in = self.dim_in[-1:0:-1]
        self.dim_out = args.enc_dim_out
        self.encoder = args.encoder
        self.pretrained = args.pretrained

        self.num_lateral_stages = len(self.dim_in)
        self.topdown_lateral_modules = nn.ModuleList()

        for i in range(self.num_lateral_stages):
            self.topdown_lateral_modules.append(
                lateral_block(self.dim_in[i], self.dim_out[i]))

        self.bottomup = conv_body_func(args.freeze_backbone)
        dilation_rate = [4, 8, 12] if 'stride_8' in self.encoder else [2, 4, 6]
        encoder_stride = 8 if 'stride8' in self.encoder else 16
        if 'mobilenetv2' in self.encoder:
            self.bottomup_top = Global_pool_block(self.dim_in[0], self.dim_out[0], encoder_stride, args.crop_size)
        else:
            self.bottomup_top = ASPP_block(self.dim_in[0], self.dim_out[0], dilation_rate, encoder_stride)

        self._init_modules(args.init_type)

    def _init_modules(self, init_type):
        if self.pretrained:
            load_pretrained_imagenet_weights(self.bottomup, self.encoder)

        self._init_weights(init_type)

    def _init_weights(self, init_type='xavier'):
        def init_func(m):
            if isinstance(m, nn.Conv2d):
                if init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                if init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)
                if init_type == 'gaussian':
                    nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)
        def init_model_weight(m):
            for child_m in m.children():
                if not isinstance(child_m, nn.ModuleList):
                    child_m.apply(init_func)

        if self.pretrained:
            init_model_weight(self.topdown_lateral_modules)
            init_model_weight(self.bottomup_top)
        else:
            init_model_weight(self)

    def forward(self, x):
        _, _, h, w = x.shape
        backbone_stage_size = [(math.ceil(h/(2.0**i)), math.ceil(w/(2.0**i))) for i in range(5, 0, -1)]
        backbone_stage_size.append((h, w))
        bottemup_blocks_out = [self.bottomup.res1(x)]
        for i in range(1, self.bottomup.convX):
            bottemup_blocks_out.append(
                getattr(self.bottomup, 'res%d' % (i + 1))(bottemup_blocks_out[-1])
            )
        bottemup_top_out = self.bottomup_top(bottemup_blocks_out[-1])
        lateral_blocks_out = [bottemup_top_out]
        for i in range(self.num_lateral_stages):
            lateral_blocks_out.append(self.topdown_lateral_modules[i](
                bottemup_blocks_out[-(i + 1)]
            ))
        return lateral_blocks_out, backbone_stage_size

class Global_pool_block(nn.Module):
    def __init__(self, dim_in, dim_out, output_stride, crop_size):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.globalpool_conv1x1 = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=1, padding=0, bias=False)
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.globalpool_bn = nn.BatchNorm2d(self.dim_out, momentum=0.9)
        self.unpool = nn.AdaptiveAvgPool2d((int(crop_size[0] / output_stride), int(crop_size[1] / output_stride)))

    def forward(self, x):
        out = self.globalpool_conv1x1(x)
        out = self.globalpool_bn(out)
        out = self.globalpool(out)
        out = self.unpool(out)
        return out

class ASPP_block(nn.Module):
    def __init__(self, dim_in, dim_out, dilate_rates, output_stride):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dilate_rates = dilate_rates
        self.aspp_conv1x1 = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=1, padding=0, bias=False)
        self.aspp_conv3_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=self.dilate_rates[0],
                                      dilation=self.dilate_rates[0], bias=False)
        self.aspp_conv3_2 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=self.dilate_rates[1],
                                      dilation=self.dilate_rates[1], bias=False)
        self.aspp_conv3_3 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=self.dilate_rates[2],
                                      dilation=self.dilate_rates[2], bias=False)
        self.aspp_bn1x1 = nn.BatchNorm2d(self.dim_out, momentum=0.5)
        self.aspp_bn3_1 = nn.BatchNorm2d(self.dim_out, momentum=0.5)
        self.aspp_bn3_2 = nn.BatchNorm2d(self.dim_out, momentum=0.5)
        self.aspp_bn3_3 = nn.BatchNorm2d(self.dim_out, momentum=0.5)

        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.globalpool_conv1x1 = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=1, padding=0, bias=False)
        self.globalpool_bn = nn.BatchNorm2d(self.dim_out, momentum=0.5)

    def forward(self, x):
        x1 = self.aspp_conv1x1(x)
        x1 = self.aspp_bn1x1(x1)
        x2 = self.aspp_conv3_1(x)
        x2 = self.aspp_bn3_1(x2)
        x3 = self.aspp_conv3_2(x)
        x3 = self.aspp_bn3_2(x3)
        x4 = self.aspp_conv3_3(x)        
        x4 = self.aspp_bn3_3(x4)
        
        x5 = self.globalpool(x)
        x5 = self.globalpool_conv1x1(x5)
        x5 = self.globalpool_bn(x5) # problem with bs = 1 !!
        w, h = x1.size(2), x1.size(3)
        x5 = F.interpolate(input=x5, size=(w, h), mode='bilinear', align_corners=True)

        out = torch.cat([x1, x2, x3, x4, x5], 1)
        return out


class lateral_block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.lateral = FTB_block(dim_in, dim_out)

    def forward(self, x):
        out = self.lateral(x)
        return out

class fcn_topdown(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dim_in = args.dec_dim_in
        self.dim_out = args.dec_dim_out + [args.dec_out_c]

        self.num_fcn_topdown = len(self.dim_in)
        aspp_blocks_num = 1 if 'mobilenetv2' in args.encoder else 5
        self.top = nn.Sequential(
            nn.Conv2d(self.dim_in[0] * aspp_blocks_num, self.dim_in[0], 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.dim_in[0], 0.5)
        )
        self.topdown_fcn1 = fcn_topdown_block(self.dim_in[0], self.dim_out[0])
        self.topdown_fcn2 = fcn_topdown_block(self.dim_in[1], self.dim_out[1])
        self.topdown_fcn3 = fcn_topdown_block(self.dim_in[2], self.dim_out[2])
        self.topdown_fcn4 = fcn_topdown_block(self.dim_in[3], self.dim_out[3])
        self.topdown_fcn5 = fcn_last_block(self.dim_in[4], self.dim_out[4])
        self.topdown_predict = fcn_topdown_predict(self.dim_in[5], self.dim_out[5])

        self.init_type = args.init_type
        self._init_modules(self.init_type)

    def _init_modules(self, init_type):
        self._init_weights(init_type)

    def _init_weights(self, init_type='xavier'):
        def init_func(m):
            if isinstance(m, nn.Conv2d):
                if init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                if init_type == 'kaiming':
                    nn.init.kaiming_normal(m.weight)
                if init_type == 'gaussian':
                    nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0.0, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

        for child_m in self.children():
            child_m.apply(init_func)

    def forward(self, laterals, backbone_stage_size):
        x = self.top(laterals[0])
        x1 = self.topdown_fcn1(laterals[1], x)
        x2 = self.topdown_fcn2(laterals[2], x1)
        x3 = self.topdown_fcn3(laterals[3], x2)
        x4 = self.topdown_fcn4(laterals[4], x3)
        x5 = self.topdown_fcn5(x4, backbone_stage_size)
        x6 = self.topdown_predict(x5)
        return x6


class fcn_topdown_block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.afa_block = AFA_block(dim_in)
        self.ftb_block = FTB_block(self.dim_in, self.dim_out)

    def forward(self, lateral, top, size=None):
        if lateral.shape != top.shape:
            h, w = lateral.size(2), lateral.size(3)
            top = F.interpolate(input=top, size=(h, w), mode='bilinear',align_corners=True)
        out = self.afa_block(lateral, top)
        out = self.ftb_block(out)
        return out


class fcn_topdown_predict(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = nn.Dropout2d(0.0)
        self.conv1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=2, dilation=2, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv1(x)
        x_softmax = self.softmax(x)
        return x, x_softmax


class FTB_block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.conv1 = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=2, dilation=2, bias=True)
        self.bn1 = nn.BatchNorm2d(self.dim_out, momentum=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=2, dilation=2, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        out = self.conv2(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        out = self.relu(out)
        return out


class AFA_block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim_in = dim * 2
        self.dim_out = dim
        self.dim_mid = int(dim / 8)
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(self.dim_in, self.dim_mid, 1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.dim_mid, self.dim_out, 1, stride=1, padding=0, bias=False)
        self.sigmd = nn.Sigmoid()

    def forward(self, lateral, top):
        w = torch.cat([lateral, top], 1)
        w = self.globalpool(w)
        w = self.conv1(w)
        w = self.relu(w)
        w = self.conv2(w)
        w = self.sigmd(w)
        out = w * lateral + top
        return out


class fcn_last_block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ftb = FTB_block(dim_in, dim_out)

    def forward(self, input, backbone_stage_size):
        out = F.interpolate(input=input, size=(backbone_stage_size[4][0], backbone_stage_size[4][1]), mode='bilinear', align_corners=True)
        out = self.ftb(out)
        out = F.interpolate(input=out, size=(backbone_stage_size[5][0], backbone_stage_size[5][1]), mode='bilinear', align_corners=True)
        return out

def MobileNetV2_body():
    return MobileNetV2()

def MobileNetV2_body_stride16(val):
    return MobileNetV2(output_stride=16)

def MobileNetV2_body_stride8(val):
    return MobileNetV2(output_stride=8)


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, groups=hidden_dim, bias=False, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, groups=hidden_dim, bias=False, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            out = self.conv(x)
            out += x
            return out
        else:
            return self.conv(x)

def add_block(res_setting, input_channel, width_mult=1, dilation=1):
    # building inverted residual blocks
    block = []
    for t, c, n, s in res_setting:
        output_channel = int(c * width_mult)
        for i in range(n):
            if i == 0:
                block.append(InvertedResidual(input_channel, output_channel, s, expand_ratio=t, dilation=dilation))
            else:
                block.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio=t, dilation=dilation))
            input_channel = output_channel
    return nn.Sequential(*block), output_channel


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1., output_stride=32):
        super(MobileNetV2, self).__init__()
        input_channel = 32
        last_channel = 320
        self.convX = 5
        stride1 = 1 if 32 / output_stride == 4 else 2
        stride2 = 1 if 32 / output_stride > 1 else 2
        dilation1 = 1 if stride1 == 2 else 2
        dilation2 = 1 if stride2 == 2 else (2 if stride1 == 2 else 4)

        interverted_residual_setting_block2 = [
             #t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
        ]
        interverted_residual_setting_block3 = [
            # t, c, n, s
            [6, 32, 3, 2],
        ]
        interverted_residual_setting_block4 = [
            # t, c, n, s
            [6, 64, 4, stride1],
            [6, 96, 3, 1],
        ]
        interverted_residual_setting_block5 = [
            # t, c, n, s
            [6, 160, 3, stride2],
            [6, 320, 1, 1],
        ]


        # building first layer
        #assert cfg.CROP_SIZE[0] % 32 == 0 and cfg.CROP_SIZE[1] % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = last_channel
        self.res1 = nn.Sequential(conv_bn(3, input_channel, 2))

        self.res2, output_channel = add_block(interverted_residual_setting_block2, input_channel, width_mult)

        self.res3, output_channel = add_block(interverted_residual_setting_block3, output_channel, width_mult)

        self.res4, output_channel = add_block(interverted_residual_setting_block4, output_channel, width_mult, dilation1)

        self.res5, output_channel = add_block(interverted_residual_setting_block5, output_channel, width_mult, dilation2)

        self._initialize_weights()

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'res%d' % (i + 1))(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def ResNeXt50_32x4d_body_stride16(freeze_backbone):
    return ResNeXt_body((3, 4, 6, 3), 32, 4, 16, freeze_backbone)


def ResNeXt101_32x4d_body_stride16(freeze_backbone):
    return ResNeXt_body((3, 4, 23, 3), 32, 4, 16, freeze_backbone)


class ResNeXt_body(nn.Module):
    def __init__(self, block_counts, cardinality, base_width, output_stride, freeze_backbone):
        super().__init__()
        self.block_counts = block_counts
        self.convX = len(block_counts) + 1
        self.num_layers = (sum(block_counts) + 3 * (self.convX == 4)) * 3 + 2
        self.freeze_backbone = freeze_backbone

        self.res1 = basic_bn_stem()
        dim_in = 64
        res5_dilate = int(32 / output_stride)
        res5_stride = 2 if res5_dilate == 1 else 1
        res4_dilate = 1 if res5_dilate <= 2 else 2
        res4_stride = 2 if res4_dilate == 1 else 1

        self.res2, dim_in = add_stage(dim_in, 256, block_counts[0], cardinality, base_width,
                                      dilation=1, stride_init=1)
        self.res3, dim_in = add_stage(dim_in, 512, block_counts[1], cardinality, base_width,
                                      dilation=1, stride_init=2)
        self.res4, dim_in = add_stage(dim_in, 1024, block_counts[2], cardinality, base_width,
                                      dilation=res4_dilate, stride_init=res4_stride)
        self.res5, dim_in = add_stage(dim_in, 2048, block_counts[3], cardinality, base_width,
                                      dilation=res5_dilate, stride_init=res5_stride)
        self.spatial_scale = 1 / output_stride
        self.dim_out = dim_in
        self._init_modle()

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'res%d' % (i + 1))(x)
        return x


    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(1, self.convX + 1):
            getattr(self, 'res%d' % i).train(mode)
    def _init_modle(self):
        def freeze_params(m):
            for p in m.parameters():
                p.requires_grad = False
        if self.freeze_backbone:
            self.apply(lambda m: freeze_params(m) if isinstance(m, nn.BatchNorm2d) else None)

def basic_bn_stem():
    conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
    return nn.Sequential(OrderedDict([
        ('conv1', conv1),
        ('bn1', nn.BatchNorm2d(64)),
        ('relu', nn.ReLU(inplace=True)),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))

def add_stage(inplanes, outplanes, nblocks, cardinality, base_width, dilation=1, stride_init=2):
    """Make a stage consist of `nblocks` residual blocks.
    Returns:
        - stage module: an nn.Sequentail module of residual blocks
        - final output dimension
    """
    res_blocks = []
    stride = stride_init
    for _ in range(nblocks):
        res_blocks.append(ResNeXtBottleneck(
            inplanes, outplanes, stride, dilation, cardinality, base_width
        ))
        inplanes = outplanes
        stride = 1
    return nn.Sequential(*res_blocks), outplanes


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, dilate, cardinality=32, base_width=4):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / 256.
        D = cardinality * base_width * int(width_ratio)
        self.conv1 = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D)
        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=dilate, dilation=dilate, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(D)
        self.conv3 = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential()
            self.shortcut.add_module('conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

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

        if self.shortcut is not None:
            residual = self.shortcut(x)

        out += residual
        out = self.relu(out)
        return out


class MetricDepthModel(nn.Module):
    def __init__(self, args):
        super(MetricDepthModel, self).__init__()
        self.loss_names = ['Weighted_Cross_Entropy', 'Virtual_Normal']
        self.depth_model = DepthModel(args)

    def forward(self, x):
        # Input data is a_real, predicted data is b_fake, groundtruth is b_real
        self.a_real = x
        self.b_fake_logit, self.b_fake_softmax = self.depth_model(self.a_real)
        return self.b_fake_logit, self.b_fake_softmax

class DepthModel(nn.Module):
    def __init__(self, args):
        super(DepthModel, self).__init__()
        self.encoder_modules = get_func(args)
        self.decoder_modules = fcn_topdown(args)

    def forward(self, x):
        lateral_out, encoder_stage_size = self.encoder_modules(x)
        out_logit, out_softmax = self.decoder_modules(lateral_out, encoder_stage_size)
        return out_logit, out_softmax


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default='resnext101_32x4d_body_stride16', type=str, help='Encoder architecture')
    parser.add_argument('--init_type', default='xavier', type=str, help='Weight initialization')
    parser.add_argument('--pretrained', action='store_true', help='pretrained backbone')
    parser.add_argument('--enc_dim_in', nargs='+', default=[64, 256, 512, 1024, 2048], help='encoder input features')
    parser.add_argument('--enc_dim_out', nargs='+', default=[512, 256, 256, 256], help='encoder output features')
    parser.add_argument('--dec_dim_in', nargs='+', default=[512, 256, 256, 256, 256, 256], help='decoder input features')
    parser.add_argument('--dec_dim_out', nargs='+', default=[256, 256, 256, 256, 256], help='decoder output features')
    parser.add_argument('--dec_out_c', default=150, type=int, help='decoder output channels')
    parser.add_argument('--crop_size', default=(385, 385), help='Crop size for mobilenet')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone')

    args = parser.parse_args()

    model = torchvision.models.resnext50_32x4d(pretrained=True)
    print(dict(model.named_parameters()).keys())