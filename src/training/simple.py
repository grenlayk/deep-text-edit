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
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            storage: Storage,
            logger: Logger,
            max_epoch: int, 
            device: str):
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

        for input, target in self.train_dataloader():
            input = input.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(input)
            loss = self.criterion(pred, target)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            self.logger.log_train(losses={'main': loss.item()}, images={'input': input, 'output': pred, 'target': target})

        
    def validate(self, epoch):
        self.model.eval()

        for input, target in self.val_dataloader():
            input = input.to(self.device)
            target = target.to(self.device)

            pred = self.model(input)
            loss = self.criterion(pred, target)
            metric = metric(pred, target)
            self.logger.log_val(losses={'main': loss.item()}, metrics={'metric': metric}, 
                                images={'input': input, 'output': pred, 'target': target})

        avg_losses, avg_metrics = self.logger.end_val()
        self.storage.save(epoch, {'model': self.model, 'optimizer': self.optimizer, 'scheduler': self.scheduler}, avg_metrics['metric'])

    def run(self):
        for epoch in range(self.max_epoch):
            self.train()
            with torch.no_grad():
                self.validate(epoch)
