import warnings
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class WarmupScheduler(_LRScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            warmup_epochs: int,
            scheduler: _LRScheduler,
            last_epoch=-1,
            verbose=False):
        self.optimizer = optimizer
        self.end_lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.warmup_epochs = warmup_epochs
        self.scheduler = scheduler
        super().__init__(optimizer=optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch > self.warmup_epochs:
            return self.scheduler.get_lr()
        else:
            return [end_lr / self.warmup_epochs * self.last_epoch
                    for end_lr in self.end_lrs]

    def step(self, epoch=None):
        if self.last_epoch > self.warmup_epochs:
            self.scheduler.step(self.last_epoch - self.warmup_epochs)
        super().step(epoch=epoch)
