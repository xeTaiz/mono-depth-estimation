import torch.nn as nn
from torchvision.models.resnet import resnet50

class ResidualBackend(nn.Module):
    def __init__(self, n_filter, pretrained=False):
        ''' Residual feature extractor

        Args:
            n_filter (int): Number of filters
            latent_sz (int): Size of the resulting image feature vector
        '''
        super().__init__()
        resnet = resnet50(pretrained=pretrained)
        self.layer1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
        )
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x