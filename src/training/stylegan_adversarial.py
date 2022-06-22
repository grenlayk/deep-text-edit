import torch
from src.logger.simple import Logger
from src.storage.simple import Storage
from src.utils.draw import draw_word, img_to_tensor
from torch import nn, optim
from torch.utils.data import DataLoader
from loguru import logger



class StyleGanAdvTrainer:
    def __init__(self,
                 model_G: nn.Module,
                 model_D: nn.Module,
                 style_embedder: nn.Module,
                 content_embedder: nn.Module,
                 optimizer_G: optim.Optimizer,
                 optimizer_D: optim.Optimizer,
                 scheduler_G: optim.lr_scheduler._LRScheduler,
                 scheduler_D: optim.lr_scheduler._LRScheduler,
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
                 adv_coef: float,
                 ocr_loss: nn.Module,
                 typeface_loss: nn.Module,
                 perc_loss: nn.Module,
                 cons_loss: nn.Module,
                 adv_loss: nn.Module
                 ):

        self.device = device
        self.model_G = model_G
        self.model_D = model_D
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.scheduler_G = scheduler_G
        self.scheduler_D = scheduler_D
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
        self.adv_coef = adv_coef
        self.typeface_loss = typeface_loss.to(device)
        self.ocr_loss = ocr_loss.to(device)
        self.perc_loss = perc_loss.to(device)
        self.cons_loss = cons_loss.to(device)
        self.adv_loss = adv_loss.to(device)
        self.style_embedder = style_embedder
        self.content_embedder = content_embedder
    
    def set_requires_grad(self, net: nn.Module, requires_grad: bool = False):
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
    
    def model_D_adv_loss(self, 
                         style_imgs: torch.Tensor, 
                         content_embeds: torch.Tensor, 
                         style_embeds: torch.Tensor
                         ):
        pred_D_fake = self.model_D(self.model_G(content_embeds, style_embeds).detach())
        pred_D_real = self.model_D(style_imgs)
        fake = torch.tensor(0.).expand_as(pred_D_fake).to(self.device)
        real = torch.tensor(1.).expand_as(pred_D_real).to(self.device)
        return (self.adv_loss(pred_D_real, real) + self.adv_loss(pred_D_fake, fake)) / 2.

    def model_G_adv_loss(self, preds: torch.Tensor):
        pred_D_fake = self.model_D(preds)
        valid = torch.tensor(1.).expand_as(pred_D_fake).to(self.device)
        return self.adv_loss(pred_D_fake, valid)

    def train(self):
        logger.info('Start training')
        self.model_G.train()
        self.model_D.train()
        self.content_embedder.train()
        self.style_embedder.train()

        # style_imgs - images containing the style we want to imitate
        # desired_content - rendered images containing the content we want to draw
        # desired_labels - text labels of content batch
        # style_content - rendered images containing the words from the style images
        # style_labels - text labels of the style_imgs batch
        for style_imgs, desired_content, desired_labels, style_content, style_labels in self.train_dataloader:
            if max(len(label) for label in desired_labels) > 25:
                continue
            if max(len(label) for label in style_labels) > 25:
                continue
            
            self.optimizer_G.zero_grad()
            self.optimizer_D.zero_grad()

            style_imgs = style_imgs.to(self.device)
            desired_content = desired_content.to(self.device)
            style_content = style_content.to(self.device)

            style_embeds = self.style_embedder(style_imgs)
            content_embeds = self.content_embedder(desired_content)
            style_content_embeds = self.content_embedder(style_content)

            ### calculate D loss
            self.set_requires_grad(self.model_D, True)
            self.set_requires_grad(self.model_G, False) 
            loss_D = self.model_D_adv_loss(style_imgs, style_content_embeds, style_embeds)

            ### calculate G losses
            self.set_requires_grad(self.model_D, False)
            self.set_requires_grad(self.model_G, True) 
            
            preds = self.model_G(content_embeds, style_embeds)
            ocr_loss_preds, recognized = self.ocr_loss(preds, desired_labels, return_recognized=True)
            word_images = torch.stack(list(map(lambda word: img_to_tensor(draw_word(word)), recognized)))

            reconstructed = self.model_G(style_content_embeds, style_embeds)
            reconstructed_loss = self.cons_loss(style_imgs, reconstructed)

            reconstructed_style_embeds = self.style_embedder(reconstructed)
            cycled = self.model_G(style_content_embeds, reconstructed_style_embeds)
            cycle_loss = self.cons_loss(style_imgs, cycled)

            ocr_loss_rec = self.ocr_loss(reconstructed, style_labels)
            ocr_loss = (ocr_loss_preds + ocr_loss_rec) / 2.

            perc_loss, tex_loss = self.perc_loss(style_imgs, reconstructed)
            emb_loss = self.typeface_loss(style_imgs, reconstructed)

            adv_loss = self.model_G_adv_loss(reconstructed)

            loss_G = \
                self.ocr_coef * ocr_loss + \
                self.cycle_coef * cycle_loss + \
                self.recon_coef * reconstructed_loss + \
                self.perc_coef * perc_loss + \
                self.tex_coef * tex_loss + \
                self.emb_coef * emb_loss + \
                self.adv_coef * adv_loss

            # update models
            self.set_requires_grad(self.model_D, True)

            loss_G.backward()
            loss_D.backward()

            self.optimizer_D.step() 
            self.optimizer_G.step()

            self.logger.log_train(
                losses={
                    'ocr_loss': ocr_loss.item(),
                    'perc_loss': perc_loss.item(),
                    'cycle_loss': cycle_loss.item(),
                    'recon_loss': reconstructed_loss.item(),
                    'tex_loss': tex_loss.item(),
                    'emb_loss': emb_loss.item(),
                    'adv_loss': adv_loss.item(),
                    'disc_loss': loss_D.item(),
                    'full_loss': loss_G.item()},
                images={
                    'style': style_imgs,
                    'content': desired_content,
                    'reconstructed': reconstructed,
                    'result': preds,
                    'recognized': word_images})

    def validate(self, epoch: int):
        self.model_G.eval()
        self.model_D.eval()
        self.content_embedder.eval()
        self.style_embedder.eval()

        for style_imgs, desired_content, desired_labels, style_content, style_labels in self.val_dataloader:
            if max(len(label) for label in desired_labels) > 25:
                continue
            if max(len(label) for label in style_labels) > 25:
                continue

            self.optimizer_G.zero_grad()
            self.optimizer_D.zero_grad()

            style_imgs = style_imgs.to(self.device)
            desired_content = desired_content.to(self.device)
            style_content = style_content.to(self.device)
            style_embeds = self.style_embedder(style_imgs)
            content_embeds = self.content_embedder(desired_content)

            preds = self.model_G(content_embeds, style_embeds)
            ocr_loss_preds = self.ocr_loss(preds, desired_labels)

            style_label_embeds = self.content_embedder(style_content)

            reconstructed = self.model_G(style_label_embeds, style_embeds)
            reconstructed_loss = self.cons_loss(style_imgs, reconstructed)

            reconstructed_style_embeds = self.style_embedder(reconstructed)
            cycle = self.model_G(style_label_embeds, reconstructed_style_embeds)
            cycle_loss = self.cons_loss(style_imgs, cycle)

            ocr_loss_rec = self.ocr_loss(reconstructed, style_labels)
            ocr_loss = (ocr_loss_preds + ocr_loss_rec) / 2.

            perc_loss, tex_loss = self.perc_loss(style_imgs, preds)
            emb_loss = self.typeface_loss(style_imgs, preds)
            adv_loss = self.model_G_adv_loss(reconstructed)

            loss = \
                self.ocr_coef * ocr_loss + \
                self.cycle_coef * cycle_loss + \
                self.recon_coef * reconstructed_loss + \
                self.perc_coef * perc_loss + \
                self.tex_coef * tex_loss + \
                self.emb_coef * emb_loss + \
                self.adv_coef * adv_loss

            self.logger.log_val(
                losses={
                    'ocr_loss': ocr_loss.item(),
                    'perc_loss': perc_loss.item(),
                    'cycle_loss': cycle_loss.item(),
                    'recon_loss': reconstructed_loss.item(),
                    'tex_loss': tex_loss.item(),
                    'emb_loss': emb_loss.item(),
                    'adv_loss': adv_loss.item(),
                    'full_loss': loss.item()},
                images={
                    'style': style_imgs,
                    'content': desired_content,
                    'result': preds})

        avg_losses, _ = self.logger.end_val()
        self.storage.save(epoch,
                          {'model_G': self.model_G,
                           'model_D': self.model_D,
                           'content_embedder': self.content_embedder,
                           'style_embedder': self.style_embedder,
                           'optimizer_G': self.optimizer_G,
                           'optimizer_D': self.optimizer_D,
                           'scheduler_G': self.scheduler_G,
                           'scheduler_D': self.scheduler_D},
                          avg_losses['full_loss'])

    def run(self):
        for epoch in range(self.total_epochs):
            self.train()
            with torch.no_grad():
                self.validate(epoch)
            if self.scheduler_G is not None:
                self.scheduler_G.step()
            if self.scheduler_D is not None:
                self.scheduler_D.step()
