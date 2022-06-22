import torch
from src.logger.simple import Logger
from src.storage.simple import Storage
from torch import nn, optim
from torch.utils.data import DataLoader
from loguru import logger

from src.utils.draw import draw_word, img_to_tensor


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
                 ocr_coef: float,
                 cycle_coef: float,
                 recon_coef: float,
                 emb_coef: float,
                 perc_coef: float,
                 tex_coef: float,
                 ocr_loss: nn.Module,
                 typeface_loss: nn.Module,
                 perc_loss: nn.Module,
                 cons_loss: nn.Module
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
        self.ocr_coef = ocr_coef
        self.cycle_coef = cycle_coef
        self.recon_coef = recon_coef
        self.emb_coef = emb_coef
        self.perc_coef = perc_coef
        self.tex_coef = tex_coef
        self.typeface_loss = typeface_loss.to(device)
        self.ocr_loss = ocr_loss.to(device)
        self.perc_loss = perc_loss.to(device)
        self.cons_loss = cons_loss.to(device)
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
            ocr_loss, recognized = self.ocr_loss(preds, desired_labels, return_recognized=True)
            word_images = torch.stack(list(map(lambda word: img_to_tensor(draw_word(word)), recognized)))

            style_content_embeds = self.content_embedder(style_content)

            reconstructed = self.model(style_content_embeds, style_embeds)
            reconstructed_loss = self.cons_loss(style_imgs, reconstructed)

            reconstructed_style_embeds = self.style_embedder(reconstructed)
            cycle = self.model(style_content_embeds, reconstructed_style_embeds)
            cycle_loss = self.cons_loss(style_imgs, cycle)

            perc_loss, tex_loss = self.perc_loss(style_imgs, reconstructed)
            emb_loss = self.typeface_loss(style_imgs, reconstructed)

            loss = \
                self.ocr_coef * ocr_loss + \
                self.cycle_coef * cycle_loss + \
                self.recon_coef * reconstructed_loss + \
                self.perc_coef * perc_loss + \
                self.tex_coef * tex_loss + \
                self.emb_coef * emb_loss

            loss.backward()
            self.optimizer.step()

            self.logger.log_train(
                losses={
                    'ocr_loss': ocr_loss.item(),
                    'perc_loss': perc_loss.item(),
                    'cycle_loss': cycle_loss.item(),
                    'recon_loss': reconstructed_loss.item(),
                    'tex_loss': tex_loss.item(),
                    'emb_loss': emb_loss.item(),
                    'full_loss': loss.item()},
                images={
                    'style': style_imgs,
                    'content': desired_content,
                    'reconstructed': reconstructed,
                    'result': preds,
                    'recognized': word_images})

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
            ocr_loss = self.ocr_loss(preds, desired_labels)

            style_label_embeds = self.content_embedder(style_content)

            reconstructed = self.model(style_label_embeds, style_embeds)
            reconstructed_loss = self.cons_loss(style_imgs, reconstructed)

            reconstructed_style_embeds = self.style_embedder(reconstructed)
            cycle = self.model(style_label_embeds, reconstructed_style_embeds)
            cycle_loss = self.cons_loss(style_imgs, cycle)

            perc_loss, tex_loss = self.perc_loss(style_imgs, preds)
            emb_loss = self.typeface_loss(style_imgs, preds)

            loss = \
                self.ocr_coef * ocr_loss + \
                self.cycle_coef * cycle_loss + \
                self.recon_coef * reconstructed_loss + \
                self.perc_coef * perc_loss + \
                self.tex_coef * tex_loss + \
                self.emb_coef * emb_loss

            self.logger.log_val(
                losses={
                    'ocr_loss': ocr_loss.item(),
                    'perc_loss': perc_loss.item(),
                    'cycle_loss': cycle_loss.item(),
                    'recon_loss': reconstructed_loss.item(),
                    'tex_loss': tex_loss.item(),
                    'emb_loss': emb_loss.item(),
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
