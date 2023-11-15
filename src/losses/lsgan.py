import torch
from torch import nn


class LSGeneratorCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real, fake):
        return torch.mean((fake - 1) ** 2)


class LSDiscriminatorCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real, fake):
        loss_fake = torch.mean((fake - 0) ** 2)
        loss_real = torch.mean((real - 1) ** 2)
        loss = 0.5 * loss_fake + 0.5 * loss_real
        return loss
