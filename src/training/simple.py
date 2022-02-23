import torch
import torch.nn as nn 

from torch import optim
from src.models.simple import Model
from src.logger.storage import Storage
from src.logger.simple import Logger


class SimpleTrainer():
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim, 
                 storage: Storage, logger: Logger, train_dataloader: any, val_dataloader: any):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.storage = storage
        self.logger = logger
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def train(self):
        for input, target in self.train_dataloder():
            pred = self.model(input)
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()
            self.logger.log_train({'main': loss.item()}, {'input': input})

    def validate(self, epoch):
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
        self.logger.log_val({'avg_loss': avg_loss}, {'avg_metric': avg_metric}, {'input': input}) # 1 print
        self.storage.save(epoch, {}, avg_metric)

    def run(self, max_epoch=10):
        for epoch in range(max_epoch):
            self.train()
            with torch.no_grad():
                self.validate(epoch)