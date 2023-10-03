from torch import nn


class LossScaler(nn.Module):
    def __init__(self, criterion: nn.Module, scale: float):
        super().__init__()
        self.criterion = criterion
        self.scale = scale

    def forward(self, *args, **kwargs):
        loss = self.criterion(*args, **kwargs)
        return self.scale * loss
