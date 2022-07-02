# source: https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
# 
# Copyright (c) 2020 Alper Ahmetoğlu
#
# ------------------------------------------------
# modification of perceptual.py and gram.py files with normalization 

import torch
import torchvision
from torchvision import transforms as T


class VGGLoss(torch.nn.Module):
    def __init__(self, perc_layers=None, tex_layers=None):
        super().__init__()

        if perc_layers is None:
            perc_layers = [0, 1, 2, 3, 4]
        if tex_layers is None:
            tex_layers = [0, 1, 2, 3, 4]

        model = torchvision.models.vgg16(pretrained=True)
        vgg_layers = list(model.features.children())
        blocks = [
            torch.nn.Sequential(*vgg_layers[:4]).eval(),
            torch.nn.Sequential(*vgg_layers[4:9]).eval(),
            torch.nn.Sequential(*vgg_layers[9:16]).eval(),
            torch.nn.Sequential(*vgg_layers[16:23]).eval(),
            torch.nn.Sequential(*vgg_layers[23:30]).eval()
        ]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        self.transform = torch.nn.functional.interpolate
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.tex_layers = tex_layers
        self.perc_layers = perc_layers

    def forward(self, inputs, target):
        if inputs.shape[1] != 3:
            inputs = inputs.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        perc_loss = 0.0
        gram_loss = 0.0
        x = inputs
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            _, _, h, w = x.shape
            if i in self.perc_layers:
                perc_loss += torch.nn.functional.l1_loss(x, y) / (h * w) / len(self.perc_layers)
            if i in self.tex_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                gram_loss += torch.nn.functional.l1_loss(gram_x, gram_y) / (h * w) / len(self.tex_layers)
        return perc_loss, gram_loss
