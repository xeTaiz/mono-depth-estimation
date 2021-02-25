import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.activation = nn.ELU()
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.activation(x)
        x = self.bn(x)
        x = self.conv(x)
        return x

class GlobalConsitency(nn.Module):
    def __init__(self, channels, input_size=(384, 384), out_feat=64):
        super().__init__()
        self.inc = nn.Upsample(scale_factor=2)
        self.avg = nn.AdaptiveMaxPool2d((input_size[0] // 2, input_size[1] // 2))
        self.conv = Conv2d(channels, channels // 2, kernel_size=3, padding=1, stride=1)
        self.conv_final = Conv2d(channels // 2, out_feat, kernel_size=3, padding=1, stride=1)

    def forward(self, x0, x1):
        x1 = self.inc(x1)
        x0 = self.avg(x0)
        x1 = self.avg(x1)
        x = torch.cat([x0, x1], dim=1)
        x = self.conv(x)
        x = self.conv_final(x)
        return x

class Details(nn.Module):
    def __init__(self, channels, input_size=(384,384), scale=2, out_feat=64):
        super().__init__()
        self.c = int(channels / (scale * scale))
        self.shuffle = nn.PixelShuffle(scale)
        self.down       = Conv2d(self.c,      self.c *  2,  kernel_size=3, stride=2, padding=1)
        self.conv       = Conv2d(self.c *  4, self.c *  2,  kernel_size=3, stride=1, padding=1)
        self.conv2      = Conv2d(self.c *  2, self.c     ,  kernel_size=3, stride=1, padding=1)
        self.conv_final = Conv2d(self.c     ,    out_feat,  kernel_size=3, stride=1, padding=1)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x0, x1):
        x0 = self.shuffle(x0)
        x0 = self.down(x0)
        x1 = self.shuffle(x1)
        x = torch.cat([x0, x1], dim=1)
        x = self.conv(x)
        x = self.conv2(x)
        x = self.conv_final(x)
        x = self.up(x)
        return x

class Sharpness(nn.Module):
    def __init__(self, encoder_feature_sizes, out_feat=64):
        super().__init__()
        [feat0, feat1, feat2] = encoder_feature_sizes[2:5]
        self.tconv0 = nn.ConvTranspose2d(feat1,      feat1 // 2, kernel_size=4, padding=1, stride=2)
        self.tconv1 = nn.ConvTranspose2d(feat2,      feat2 // 4, kernel_size=4, padding=1, stride=2)
        self.tconv2 = nn.ConvTranspose2d(feat2 // 4, feat2 // 8, kernel_size=4, padding=1, stride=2)

        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(feat0 + feat1 // 2 + feat2 // 8, out_feat * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(out_feat * 2, out_feat, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

    def forward(self, x0, x1, x2):
        x1 = self.tconv0(x1) 
        x2 = self.tconv1(x2) 
        x2 = self.tconv2(x2)
        
        x = torch.cat([x0, x1, x2], dim=1)
        
        x = self.up0(x)
        x = self.up1(x)
        
        return x

class Weighter(nn.Module):
    def __init__(self, input_size, in_feat, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.conv = Conv2d(in_feat, in_feat // 2, kernel_size=3, stride=2, padding=1)
        self.mlp = nn.Linear(input_size[0] * input_size[1] // 16, 3)

    def forward(self, x):
        [a,b,c] = x
        
        a = self.conv(a)
        b = self.conv(b)
        c = self.conv(c)

        x = torch.cat([a, b, c], dim=1)
        x = torch.flatten(x, start_dim=2)
        x = self.mlp(x)
       
        x = torch.mean(x, dim=1)
        if torch.sum(x) < self.epsilon:
            x = torch.ones_like(x) / 3.0
        else:
            x = x / torch.sum(x)
        return x


class my_decoder(nn.Module):
    def __init__(self, input_size, encoder_feature_sizes):
        super().__init__()
        self.refine0 = FeatureFusionBlock(encoder_feature_sizes[0])
        self.refine1 = FeatureFusionBlock(encoder_feature_sizes[1])
        self.refine2 = FeatureFusionBlock(encoder_feature_sizes[2])
        self.refine3 = FeatureFusionBlock(encoder_feature_sizes[3])

        self.global_con = GlobalConsitency(encoder_feature_sizes[0] + encoder_feature_sizes[1], input_size=input_size, out_feat=64)
        self.details = Details(encoder_feature_sizes[1], out_feat=64)
        self.sharpness = Sharpness(encoder_feature_sizes, out_feat=64)

        self.weighter = Weighter(input_size=input_size, in_feat=64)    

        self.get_depth = nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(64, 1, 3, 1, 1, bias=False), nn.Sigmoid())    

    def forward(self, features):
        skip0, skip1, skip2, skip3 = features[1], features[2], features[3], features[4]
        dense_features = torch.nn.ReLU()(features[5])
        x0 = self.refine0(skip0)
        x1 = self.refine1(skip1)
        x2 = self.refine2(skip2)
        x3 = self.refine3(skip3)

        glob = self.global_con(x0, x1)
        detail = self.details(x1, x2)
        sharpness = self.sharpness(x2, x3, dense_features)

        glob_d = self.get_depth(glob)
        detail_d = self.get_depth(detail)
        sharpness_d = self.get_depth(sharpness)

        scale = self.weighter([glob, detail, sharpness])
    
        x = torch.cat([glob_d.unsqueeze(1), detail_d.unsqueeze(1), sharpness_d.unsqueeze(1)], dim=1)
        x = x * scale[:,:,None, None, None]
        x = torch.sum(x, dim=1)
        return x

class encoder(nn.Module):
    def __init__(self, version):
        super(encoder, self).__init__()
        import torchvision.models as models
        if version == 'densenet121_bts':
            self.base_model = models.densenet121(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [64, 64, 128, 256, 1024]
        elif version == 'densenet161_bts':
            self.base_model = models.densenet161(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [96, 96, 192, 384, 2208]
        elif version == 'resnet50_bts':
            self.base_model = models.resnet50(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif version == 'resnet101_bts':
            self.base_model = models.resnet101(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif version == 'resnext50_bts':
            self.base_model = models.resnext50_32x4d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif version == 'resnext101_bts':
            self.base_model = models.resnext101_32x8d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        else:
            print('Not supported encoder: {}'.format(version))

    def forward(self, x):
        features = [x]
        skip_feat = [x]
        for k, v in self.base_model._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(features[-1])
            features.append(feature)
            if any(x in k for x in self.feat_names):
                skip_feat.append(feature)
        
        return skip_feat

class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        #output = nn.functional.interpolate(
        #    output, scale_factor=2, mode="bilinear", align_corners=True
        #)

        return output

class MyModel(nn.Module):
    def __init__(self, input_size=(384, 384), encoder_version='densenet161_bts'):
        super(MyModel, self).__init__()
        self.encoder = encoder(encoder_version)
        self.decoder = my_decoder(input_size, self.encoder.feat_out_channels)

    def forward(self, x):
        skip_feat = self.encoder(x)
        return self.decoder(skip_feat)


if __name__ == "__main__":
    input_size = (384, 384)
    model = MyModel(input_size=input_size, encoder_version='resnext101_bts')
    img = torch.rand((2,3, input_size[0], input_size[1]))
    y_hat = model(img)
    print(y_hat.shape)