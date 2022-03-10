from typing import Callable

import torch
from src.logger.simple import Logger
from src.storage.simple import Storage
from torch import nn, optim
from torch.utils.data import DataLoader


class SimpleTrainer():
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
            max_epoch: int):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.storage = storage
        self.logger = logger
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_epoch = max_epoch

    def train(self):
        self.model.train()

        for input, target in self.train_dataloder():
            pred = self.model(input)
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()
            self.logger.log_train({'main': loss.item()}, {'input': input})

        self.scheduler.step()

    def validate(self, epoch):
        self.model.eval()

        losses = 0
        metrics = 0

        for input, target in self.val_dataloder():
            pred = self.model(input)
            loss = self.criterion(pred, target)
            metric = metric(pred, target)
            metrics += metric
            losses += loss

        avg_loss = losses / len(self.val_dataloder())
        avg_metric = metric / len(self.val_dataloder())

        self.logger.log_val({'avg_loss': avg_loss}, {'avg_metric': avg_metric}, {'input': input})  # 1 print
        self.storage.save(epoch, {}, avg_metric)

    def run(self):
        for epoch in range(self.max_epoch):
            self.train()
            with torch.no_grad():
                self.validate(epoch)
