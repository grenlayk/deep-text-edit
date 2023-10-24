# source: https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html

import torch
from torch import nn
from torchvision import models


class ContentResnet(nn.Module):
    def __init__(self, resnet: models.ResNet):
        super().__init__()
        self.internal = resnet

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.internal.conv1(x)
        x = self.internal.bn1(x)
        x = self.internal.relu(x)
        x = self.internal.maxpool(x)

        x = self.internal.layer1(x)
        x = self.internal.layer2(x)
        x = self.internal.layer3(x)
        x = self.internal.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x


class StyleResnet(nn.Module):
    def __init__(self, resnet: models.ResNet):
        super().__init__()
        self.internal = resnet
        self.internal.fc = torch.nn.Identity()

    def forward(self, x):
        return super().forward(x)
