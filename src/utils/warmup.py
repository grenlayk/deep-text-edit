from typing import Optional

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


class WarmupScheduler(LambdaLR):
    def __init__(self, optimizer: Optimizer, num_steps: int, last_epoch: int = -1):
        self.num_steps = num_steps
        super().__init__(optimizer, lambda x: min(x / self.num_steps, 1.0), last_epoch=last_epoch)

    def step(self, epoch: Optional[int] = ...) -> None:
        print(f'scheduler step: {epoch}, {self.get_lr()}')
        super().step(epoch)
