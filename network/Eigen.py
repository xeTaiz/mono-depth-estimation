import torch.nn as nn
import torch
import torchvision

class Eigen(nn.Module):
    def __init__(self, scale1='vgg', pretrained=True):
        super(Eigen, self).__init__()
        if scale1 == 'vgg':
            self.scale1 = VGG(pretrained=pretrained)

        self.scale2 = Scale2()
        self.scale3 = Scale3()

    def forward(self, img):
        x0 = self.scale1(img)
        x1 = self.scale2((img, x0))
        x = self.scale3((img, x1))
        return x

class Scale2(nn.Module):
    def __init__(self):
        super(Scale2, self).__init__()
        self.conv = nn.Conv2d(3, 96, kernel_size=9, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.scale2_onestack = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=5, stride=1, padding=2, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, padding_mode='zeros'),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=5, padding=2, padding_mode='zeros', stride=2)
        )

    def forward(self, x):
        (img, out_scale1) = x
        x = self.conv(img)
        x = self.relu(x)
        x = self.pool(x)[:, :, 1:-1, 1:-1]
        x = torch.cat([x, out_scale1], dim=1)
        x = self.scale2_onestack(x)
        return x

class Scale3(nn.Module):
    def __init__(self):
        super(Scale3, self).__init__()
        self.conv = nn.Conv2d(3, 96, kernel_size=9, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.scale3_onestack = nn.Sequential(
            nn.Conv2d(97, 64, kernel_size=5, padding=2, padding_mode="zeros"),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2, padding_mode="zeros"),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2, padding_mode="zeros"),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=5, padding=2, padding_mode="zeros"),
            nn.ReLU()
        )

    def forward(self, x):
        (img, scale2_out) = x
        x = self.conv(img)[:,:, 2:-3, 2:-3]
        x = self.relu(x)
        x = self.pool(x)
        x = torch.cat([x, scale2_out], dim=1)
        x = self.scale3_onestack(x)
        return x

class VGG(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG, self).__init__()
        pretrained_model = torchvision.models.vgg19_bn(pretrained=pretrained)
        self.feature_extractor = pretrained_model._modules['features']
        self.flatten = nn.Flatten()
        self.mlp1 = nn.Linear(512 * 10 * 7, 4096)
        self.mlp2 = nn.Linear(4096, 64 * 19 * 14)
        self.upsample = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=4) # (75, 55)
        # clear memory
        del pretrained_model

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = torch.reshape(x, (-1, 64, 14, 19))
        x = self.upsample(x)
        return x


            
if __name__ == "__main__":
    eigen = Eigen()
    img = torch.rand((10,3, 240, 320), dtype=torch.float32)
    x = eigen(img)
    print(x.shape)