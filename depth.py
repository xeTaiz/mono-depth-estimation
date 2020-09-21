import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from backend import ResidualBackend
from common import Down, Up, DoubleConv, OutConv, Reshape, UpProj

class BaseLine(nn.Module):
    def __init__(self, output_size, n_filter, pretrained):
        super().__init__()
        self.output_size = output_size
        self.backend = ResidualBackend(n_filter=n_filter, pretrained=pretrained)
        num_channels = 1024
        self.conv2 = nn.Conv2d(num_channels, num_channels // 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels // 2)

        self.upSample = UpProj(num_channels // 2)

        # setting bias=true doesn't improve accuracy
        self.conv3 = nn.Conv2d(num_channels // 32, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        # resnet
        x = self.backend(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # UpProj
        x = self.upSample(x)

        x = self.conv3(x)
        x = self.bilinear(x)

        return x

    def get_1x_lr_params(self):
        """
        This generator returns all the parameters of the net layer whose learning rate is 1x lr.
        """
        return self.backend.parameters()

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters of the net layer whose learning rate is 20x lr.
        """
        b = [self.conv2, self.bn2, self.upSample, self.conv3, self.bilinear]
        for j in range(len(b)):
            for k in b[j].parameters():
                if k.requires_grad:
                    yield k
        

class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits.sigmoid()

class Projection3D(nn.Module):
    def __init__(self, n_filter=64, vol_dim=16):
        super(Projection3D, self).__init__()
        self.reshape = Reshape(-1, n_filter, vol_dim, vol_dim, vol_dim)
        self.project1 = nn.ConvTranspose3d(n_filter, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.project2 = nn.ConvTranspose3d(16,        8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.project3 = nn.ConvTranspose3d(8,         4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.project4 = nn.ConvTranspose3d(4,         1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.reshape(x)
        x = self.project1(x)
        x = self.project2(x)
        x = self.project3(x)
        x = self.project4(x)
        return x

class Projection2D(nn.Module):
    def __init__(self, n_filter=256, dim=32):
        super(Projection2D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.AdaptiveMaxPool3d((n_filter, dim, dim)),
            Reshape(-1, n_filter, dim, dim)
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(n_filter, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class DetailedDepth(nn.Module):
    def __init__(self, n_filter=32, pretrained=False):
        super(DetailedDepth, self).__init__()
        self.backend = ResidualBackend(n_filter=n_filter, pretrained=pretrained)
        self.project3d = Projection3D(n_filter=64, vol_dim=16)
        self.project2d = Projection2D(n_filter=256, dim=32)
        
    def forward(self, x):
        x = self.backend(x)
        feat_3d = self.project3d(x)
        feat_2d = self.project2d(feat_3d)
        return feat_2d, feat_3d

if __name__ == "__main__":
    model = BaseLine(output_size=(256,256), n_filter=16, pretrained=False)

    img = np.random.uniform(0,1, (4, 3, 256, 256)).astype('float32')
    img_t = torch.from_numpy(img)
    y_hat = model(img_t)
    print(y_hat.shape)