# source: https://github.com/ahemaesh/Deep-Image-Colorization
# 
# Copyright (c) 2020 Avinash H Raju, Atulya Ravishankar

import torch
from torch import nn
from torchvision.models.resnet import resnet50
from torchvision import transforms as T


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )

    def forward(self, x):
        return self.model(x)


class FusionLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        ip, emb = inputs
        emb = torch.stack([torch.stack([emb], dim=2)], dim=3)
        emb = emb.repeat(1, 1, ip.shape[2], ip.shape[3])
        fusion = torch.cat((ip, emb), 1)
        return fusion


class Decoder(nn.Module):
    def __init__(self, input_depth):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_depth, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2.0),
        )

    def forward(self, x):
        return self.model(x)


class Model(nn.Module):
    def __init__(self, depth_after_fusion, device='cpu'):
        super().__init__()
        self.device = device
        self.encoder = Encoder()
        self.fusion = FusionLayer()
        self.after_fusion = nn.Conv2d(
            in_channels=1256,
            out_channels=depth_after_fusion,
            kernel_size=1,
            stride=1,
            padding=0)
        self.bnorm = nn.BatchNorm2d(256)
        self.decoder = Decoder(depth_after_fusion)
        self.resnet = resnet50(pretrained=True, progress=True).to(self.device).eval()

        self._resnet_resize = T.Resize(300)
        self._encoder_resize = T.Resize(224)

    def forward(self, img_l):
        img_emb = self.resnet(self._resnet_resize(img_l))
        img_enc = self.encoder(self._encoder_resize(img_l))
        fusion = self.fusion([img_enc, img_emb])
        fusion = self.after_fusion(fusion)
        fusion = self.bnorm(fusion)
        return self.decoder(fusion)


def init_weights(m):
    if isinstance(m, nn.Conv2d, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
