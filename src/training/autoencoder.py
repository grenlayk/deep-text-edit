import torch
from src.logger.simple import Logger
from src.storage.simple import Storage
from torch import nn, optim
from torch.utils.data import DataLoader
from loguru import logger

from src.utils.draw import draw_word, img_to_tensor


class AutoencoderTrainer:
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
                 content_loss: nn.Module
                 ):

        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.total_epochs = total_epochs
        self.logger = logger
        self.storage = storage
        self.content_loss = content_loss.to(device)
        self.style_embedder = style_embedder
        self.content_embedder = content_embedder

    def train(self):
        logger.info('Start training')
        self.model.train()
        self.content_embedder.train()
        self.style_embedder.train()

        # style_imgs - images containing the style we want to imitate
        # desired_content - rendered images containing the content we want to draw
        # desired_labels - text labels of content batch
        # style_content - rendered images containing the words from the style images
        for style_imgs, desired_content, desired_labels, style_content in self.train_dataloader:
            if max(len(label) for label in desired_labels) > 25:
                continue

            self.optimizer.zero_grad()

            style_imgs = style_imgs.to(self.device)
            desired_content = desired_content.to(self.device)
            style_content = style_content.to(self.device)

            style_embeds = self.style_embedder(style_imgs)
            content_embeds = self.content_embedder(desired_content)

            preds = self.model(content_embeds, style_embeds)

            content_loss = self.content_loss(desired_content, preds)
            loss = content_loss

            loss.backward()
            self.optimizer.step()

            self.logger.log_train(
                losses={
                    'content_loss': content_loss.item(),
                    'full_loss': loss.item()},
                images={
                    'style': style_imgs,
                    'content': desired_content,
                    'result': preds})

    def validate(self, epoch):
        self.model.eval()
        self.content_embedder.eval()
        self.style_embedder.eval()

        for style_imgs, desired_content, desired_labels, style_content in self.val_dataloader:
            if max(len(label) for label in desired_labels) > 25:
                continue

            self.optimizer.zero_grad()

            style_imgs = style_imgs.to(self.device)
            desired_content = desired_content.to(self.device)
            style_content = style_content.to(self.device)
            style_embeds = self.style_embedder(style_imgs)
            content_embeds = self.content_embedder(desired_content)

            preds = self.model(content_embeds, style_embeds)

            content_loss = self.content_loss(desired_content, preds)
            loss = content_loss

            self.logger.log_val(
                losses={
                    'content_loss': content_loss.item(),
                    'full_loss': loss.item()},
                images={
                    'style': style_imgs,
                    'content': desired_content,
                    'result': preds})

        avg_losses, _ = self.logger.end_val()
        self.storage.save(epoch,
                          {'model': self.model,
                           'content_embedder': self.content_embedder,
                           'style_embedder': self.style_embedder,
                           'optimizer': self.optimizer,
                           'scheduler': self.scheduler},
                          avg_losses['full_loss'])

    def run(self):
        for epoch in range(self.total_epochs):
            self.train()
            with torch.no_grad():
                self.validate(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
