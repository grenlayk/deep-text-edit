from torch import nn
from torch.nn.functional import mse_loss


class LSGeneratorCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real, fake):
        return mse_loss(fake, 1)


class LSDiscriminatorCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real, fake):
        loss = 0.5 * mse_loss(fake, 0) + 0.5 * mse_loss(real, 1)
        return loss
