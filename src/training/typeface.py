from typing import Callable

import torch
from src.logger.simple import Logger
from src.storage.simple import Storage
from torch import nn, optim
from torch.utils.data import DataLoader


class TypefaceTrainer():
    def __init__(
            self,
            model: nn.Module,
            criterion: Callable,
            optimizer: optim.Optimizer,
            scheduler: optim.lr_scheduler._LRScheduler,
            storage: Storage,
            logger: Logger,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            max_epoch: int,
            device):
        self.model = model
        self.criterion = criterion
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
            self.optimizer.zero_grad()

            inputs = inputs.to(self.device)
            target = target.to(self.device)

            pred = self.model(inputs)
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()
            self.logger.log_train({'CE': loss.item()}, {'input': inputs})

        if self.scheduler is not None:
            self.scheduler.step()

    def validate(self, epoch):
        self.model.eval()

        metrics = 0

        for inputs, target in self.val_dataloader:
            inputs = inputs.to(self.device)
            target = target.to(self.device)

            pred = self.model(inputs)
            loss = self.criterion(pred, target)
            _, max_idx_class = pred.max(dim=1)
            metric = torch.mean((max_idx_class == target).float())

            metrics += metric.item()

            self.logger.log_val({'CE': loss.item()}, {'Acc': metric.item()}, {'input': inputs})

        self.logger.end_val()
        self.storage.save(epoch, {'model': self.model, 'optimizer': self.optimizer}, metrics / len(self.val_dataloader))

    def run(self):
        for epoch in range(self.max_epoch):
            self.train()
            with torch.no_grad():
                self.validate(epoch)
