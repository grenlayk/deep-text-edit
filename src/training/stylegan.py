import cv2
import numpy as np
import torch
from src.data.baseline import draw_one
from src.logger.simple import Logger
from src.storage.simple import Storage
from torch import nn, optim
from torch.utils.data import DataLoader
from loguru import logger


class StyleGanTrainer:
    def __init__(self,
                 model: nn.Module,
                 style_embedder: nn.Module,
                 content_embedder: nn.Module,
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
                 perceptual_loss: nn.Module,
                 ocr_loss: nn.Module):

        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.total_epochs = total_epochs
        self.logger = logger
        self.storage = storage
        self.ocr_loss = ocr_loss.to(device)
        self.perceptual_loss = perceptual_loss.to(device)
        self.coef_ocr = coef_ocr_loss
        self.coef_perceptual = coef_perceptual_loss
        self.style_embedder = style_embedder
        self.content_embedder = content_embedder

    def train(self):
        logger.info('Start training')
        self.model.train()
        self.content_embedder.train()
        self.style_embedder.train()

        for style_batch, content_batch, label_batch in self.train_dataloader:
            if max(len(label) for label in label_batch) > 25:
                continue

            style_batch = style_batch.to(self.device)
            content_batch = content_batch.to(self.device)
            style_embeds = self.style_embedder(style_batch)
            content_embeds = self.content_embedder(content_batch)
            self.optimizer.zero_grad()

            res = self.model(content_embeds, style_embeds)
            ocr_loss, recognized = self.ocr_loss(res, label_batch, return_recognized=True)
            words = []
            for word in recognized:
                words.append(
                    torch.from_numpy(
                        np.transpose(
                            (cv2.resize(np.array(draw_one(word)), (128, 64)) / 255)[:, :, [2, 1, 0]], (2, 0, 1)
                        )
                    ).float()
                )
            word_images = torch.stack(words)
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
                    'result': res,
                    'recognized': word_images})

    def validate(self, epoch):
        self.model.eval()
        self.content_embedder.eval()
        self.style_embedder.eval()

        for style_batch, content_batch, label_batch in self.val_dataloader:
            style_batch = style_batch.to(self.device)
            content_batch = content_batch.to(self.device)
            style_embeds = self.style_embedder(style_batch)
            content_embeds = self.content_embedder(content_batch)

            res = self.model(content_embeds, style_embeds)
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

        avg_losses, _ = self.logger.end_val()
        self.storage.save(
            epoch,
            {'model': self.model, 'optimizer': self.optimizer, 'scheduler': self.scheduler},
            avg_losses['full_loss']
        )

    def run(self):
        for epoch in range(self.total_epochs):
            self.train()
            with torch.no_grad():
                if (epoch + 3) % 10 == 0:
                    self.validate(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
