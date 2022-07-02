# source: https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
# 
# Copyright (c) 2020 Alper Ahmetoğlu
#
# ------------------------------------------------
# modification of perceptual.py file

import torch
import torchvision

class VGGGramLoss(torch.nn.Module):
    def __init__(self, resize=False, feature_layers=[], style_layers=[2, 3]):
        super(VGGGramLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.style_layers = style_layers
        self.feature_layers = feature_layers

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
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
                b, c, h, w = x.shape
                gram_x = act_x @ act_x.permute(0, 2, 1) / (h * w)
                gram_y = act_y @ act_y.permute(0, 2, 1) / (h * w)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
