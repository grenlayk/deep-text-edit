import torch
from src.logger.simple import Logger
from src.storage.simple import Storage
from torch import nn, optim
from torch.utils.data import DataLoader
from loguru import logger


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler._LRScheduler,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 storage: Storage,
                 logger: Logger,
                 total_epochs: int,
                 device: str,
                 coef_ocr_loss: float,
                 coef_perceptual_loss: float,
                 ocr_loss: nn.Module,
                 perceptual_loss: nn.Module):

        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.total_epochs = total_epochs
        self.logger = logger
        self.storage = storage
        self.ocr_loss: nn.Module = ocr_loss.to(device)
        self.perceptual_loss = perceptual_loss.to(device)
        self.coef_ocr = coef_ocr_loss
        self.coef_perceptual = coef_perceptual_loss

    def concat_batches(self, batch_1, batch_2):
        '''
        Concatenate 2 * (Bx3xHxW) image batches along channels axis -> (Bx2*3xHxW)
        '''
        return torch.cat((batch_1, batch_2), 1)

    def train(self):
        logger.info('Start training')
        self.model.train()

        for style_batch, content_batch, label_batch in self.train_dataloader:
            concat_batches = (self.concat_batches(style_batch, content_batch)).to(self.device)
            style_batch = style_batch.to(self.device)
            self.optimizer.zero_grad()

            res = self.model(concat_batches)
            ocr_loss = self.ocr_loss(res, label_batch)
            perceptual_loss = self.perceptual_loss(style_batch, res)
            loss = self.coef_ocr * ocr_loss + self.coef_perceptual * perceptual_loss
            loss.backward()
            self.optimizer.step()
            self.logger.log_train(
                losses={
                    'ocr_loss': ocr_loss.item(),
                    'perceptual_loss': perceptual_loss.item(),
                    'full_loss': loss.item()},
                images={
                    'style': style_batch,
                    'content': content_batch,
                    'result': res})

    def validate(self, epoch):
        self.model.eval()

        for style_batch, content_batch, label_batch in self.val_dataloader:
            concat_batches = (self.concat_batches(style_batch, content_batch)).to(self.device)
            style_batch = style_batch.to(self.device)
            res = self.model(concat_batches)
            ocr_loss = self.ocr_loss(res, label_batch)
            perceptual_loss = self.perceptual_loss(style_batch, res)
            loss = self.coef_ocr * ocr_loss + self.coef_perceptual * perceptual_loss

            self.logger.log_val(
                losses={
                    'ocr_loss': ocr_loss.item(),
                    'perceptual_loss': perceptual_loss.item(),
                    'full_loss': loss.item()},
                images={
                    'style': style_batch,
                    'content': content_batch,
                    'result': res})

        avg_losses, avg_metrics = self.logger.end_val()
        self.storage.save(
            epoch,
            {'model': self.model, 'optimizer': self.optimizer, 'scheduler': self.scheduler},
            avg_metrics
        )

    def run(self):
        for epoch in range(self.total_epochs):
            self.train()
            with torch.no_grad():
                self.validate(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
