import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from common import Down, Up, DoubleConv, OutConv

class BaseLine(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.encode0 = nn.Conv2d(3, 32,   kernel_size=(3,3), stride=(2,2))
        self.encode1 = nn.Conv2d(32, 64,  kernel_size=(3,3), stride=(2,2))
        self.encode2 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=(2,2))

        self.decode0 = nn.ConvTranspose2d(128, 64, kernel_size=(3,3), stride=(2,2))
        self.decode1 = nn.ConvTranspose2d(64, 32,  kernel_size=(3,3), stride=(2,2))
        self.decode2 = nn.ConvTranspose2d(32, 1,   kernel_size=(3,3), stride=(2,2))

    def forward(self, x):
        x = self.encode0(x)
        x = self.encode1(x)
        x = self.encode2(x)

        x = self.decode0(x)
        x = self.decode1(x)
        x = self.decode2(x)
        return x.sigmoid()
        

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

class DetailedDepth(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return x

if __name__ == "__main__":
    model = UNet(n_channels=3)

    img = np.random.uniform(0,1, (3, 255, 255)).astype('float32')
    img_t = torch.from_numpy(img)
    img_t = torch.unsqueeze(img_t, 0)
    y_hat = model(img_t)

    print(y_hat)