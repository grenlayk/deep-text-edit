# source: https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
# 
# Copyright (c) 2020 Alper Ahmetoğlu

import torch
import torchvision
from kornia.enhance import Normalize, Denormalize


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(
            self,
            mean,
            std,
            vgg=None,
            feature_layers=(0, 1, 2, 3),
            style_layers=(),
            resize=False
    ):
        super(VGGPerceptualLoss, self).__init__()
        if vgg is None:
            vgg = torchvision.models.vgg16(pretrained=True)
        blocks = [
            vgg.features[:4].eval(),
            vgg.features[4:9].eval(),
            vgg.features[9:16].eval(),
            vgg.features[16:23].eval(),
        ]

        self.feature_layers = feature_layers
        self.style_layers = style_layers
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.denorm = Denormalize(torch.Tensor(mean), torch.Tensor(std))

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, input, target, ):
        input = self.norm(self.denorm(input))
        target = self.norm(self.denorm(target))

        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in self.feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in self.style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
