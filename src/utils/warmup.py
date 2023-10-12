from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class WarmupScheduler(LambdaLR):
    def __init__(self, optimizer: Optimizer, num_steps: int, last_epoch: int = -1):
        self.num_steps = num_steps
        super().__init__(optimizer, lambda x: min(x / self.num_steps, 1.0), last_epoch=last_epoch)

    def step(self, epoch=None) -> None:
        print(f'\n\n\nscheduler step: {epoch} {get_lr(self.optimizer)}\n\n\n')
        super().step(epoch)
