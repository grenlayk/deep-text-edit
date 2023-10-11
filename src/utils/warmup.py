from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


class WarmupScheduler(LambdaLR):
    def __init__(self, optimizer: Optimizer, num_steps: int, last_epoch: int = -1):
        super().__init__(optimizer, lambda x: min(x / num_steps, 1.0), last_epoch=last_epoch)
