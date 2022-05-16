from typing import Callable

import torch
from src.logger.simple import Logger
from src.storage.simple import Storage
from torch import nn, optim
from torch.utils.data import DataLoader


class SimpleTrainer:
    """Base Trainer class. Intended for training an image-to-image model.
    """

    def __init__(
            self,
            model: nn.Module,
            criterion: Callable[..., torch.Tensor],
            metric: Callable[..., torch.Tensor],
            optimizer: optim.Optimizer,
            scheduler: optim.lr_scheduler._LRScheduler,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            storage: Storage,
            logger: Logger,
            max_epoch: int,
            device: str):
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.storage = storage
        self.logger = logger
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_epoch = max_epoch
        self.device = device

    def train(self):
        self.model.train()

        for inputs, target in self.train_dataloader:
            inputs = inputs.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(inputs)
            loss = self.criterion(pred, target)
            loss.backward()

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step(loss.item())

            self.logger.log_train(
                losses={'main': loss.item()},
                images={'input': inputs, 'output': pred, 'target': target}
            )

    def validate(self, epoch):
        self.model.eval()

        for inputs, target in self.val_dataloader:
            inputs = inputs.to(self.device)
            target = target.to(self.device)

            pred = self.model(inputs)
            loss = self.criterion(pred, target)
            metric = self.metric(pred, target) if self.metric is not None else None
            self.logger.log_val(
                losses={'main': loss.item()},
                metrics=metric,
                images={'input': inputs, 'output': pred, 'target': target}
            )

        _, avg_metrics = self.logger.end_val()
        self.storage.save(
            epoch,
            {'model': self.model, 'optimizer': self.optimizer, 'scheduler': self.scheduler},
            avg_metrics
        )

    def run(self):
        for epoch in range(self.max_epoch):
            self.train()
            with torch.no_grad():
                self.validate(epoch)
