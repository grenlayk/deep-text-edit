# source: https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html

import torch
from torchvision import models
from torchvision.models.resnet import BasicBlock

class ContentResnet(models.ResNet):
    def __init__(self):
        # resnet18 init
        super().__init__(BasicBlock, [2, 2, 2, 2])

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        return x


class StyleResnet(models.ResNet):
    def __init__(self):
        # resnet18 init
        super().__init__(BasicBlock, [2, 2, 2, 2])
        self.fc = torch.nn.Identity()
