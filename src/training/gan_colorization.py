# from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py
import time
import torch
from loguru import logger
from torchvision.utils import save_image


class GANColorizationTrainer:
    def __init__(self,
                 device,
                 model_G,
                 model_D,
                 criterion,
                 criterion_gan,
                 lambda_gan,
                 optimizer_G,
                 optimizer_D,
                 scheduler_G,
                 scheduler_D,
                 train_dataloader,
                 val_dataloader,
                 total_epochs,
                 logger,
                 storage):
        self.device = device
        self.model_G = model_G
        self.model_D = model_D
        self.criterion = criterion
        self.criterion_gan = criterion_gan
        self.lambda_gan = lambda_gan
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.scheduler_G = scheduler_G
        self.scheduler_D = scheduler_D
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.total_epochs = total_epochs
        self.logger = logger
        self.storage = storage

    def set_requires_grad(self, net, requires_grad=False):
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def train(self):
        self.model_G.train()
        self.model_D.train()

        # inputs: l_img x3
        # target: rgb_img
        for inputs, targets in self.train_dataloader:

            # Skip bad data
            if not inputs.ndim:
                continue

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer_G.zero_grad()
            self.optimizer_D.zero_grad()
            
            ### calculate D loss
            self.set_requires_grad(self.model_D, True)
            self.set_requires_grad(self.model_G, False) 

            # Fake
            pred_D_fake = self.model_D(self.model_G(inputs).detach())
            fake = torch.tensor(0.).expand_as(pred_D_fake).to(self.device)
            loss_D_fake = self.criterion_gan(pred_D_fake, fake)
            # Real
            pred_D_real = self.model_D(targets)
            valid = torch.tensor(1.).expand_as(pred_D_real).to(self.device)
            loss_D_real = self.criterion_gan(pred_D_real, valid) 
            loss_D = (loss_D_real + loss_D_fake) * 0.5

            ### calculate G losses
            self.set_requires_grad(self.model_D, False)
            self.set_requires_grad(self.model_G, True) 

            preds_G = self.model_G(inputs)
            pred_D_fake = self.model_D(preds_G)
            loss_G_gan = self.criterion_gan(pred_D_fake, valid)
            loss_G_sup = self.criterion(preds_G, targets)
            loss_G = self.lambda_gan * loss_G_gan + loss_G_sup['total']

            # update models
            self.set_requires_grad(self.model_D, True)

            loss = loss_D + loss_G
            loss.backward()

            self.optimizer_D.step() 
            self.optimizer_G.step()
            self.scheduler_D.step()
            self.scheduler_G.step()
            
            # Print stats after every point_batches
            self.logger.log_train(
                losses={
                'total G': loss_G.item(), 
                'G L1': loss_G_sup['L1Loss'], 
                'G Perceptual': loss_G_sup['VGGPerceptualLoss'], 
                'G GAN': self.lambda_gan * loss_G_gan.item(),
                'total D': loss_D.item(), 
                'lr G': self.scheduler_G.get_last_lr()[0],
                'lr D': self.scheduler_D.get_last_lr()[0]},
                images={
                'input': inputs, 
                'pred': torch.FloatTensor(preds_G.cpu().detach()), 
                'target': targets},
            )

    def validate(self, epoch: int):
        # Validation Step
        # Intialize Model to Eval Mode for validation
        self.model_G.eval()
        self.model_D.eval()
        for inputs, targets in self.val_dataloader:
            # Skip bad data
            if not inputs.ndim:
                continue

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            preds_G = self.model_G(inputs)
            loss_G_sup = self.criterion(preds_G, targets)

            self.logger.log_val(
                losses={
                    'G L1': loss_G_sup['L1Loss'], 
                    'G Perceptual': loss_G_sup['VGGPerceptualLoss']},
                images={
                    'val_input': inputs, 
                    'val_pred': torch.FloatTensor(preds_G.cpu().detach()), 
                    'val_target': targets},
            )

        return self.logger.end_val()

    def run(self):
        for epoch in range(self.total_epochs):
            start = time.time_ns()
            self.train()
            logger.success(f'Finished training epoch #{epoch} in {(time.time_ns() - start) / 1e9}s')

            with torch.no_grad():
                _, _ = self.validate(epoch)

            # Save the Model to disk
            self.storage.save(
                epoch=epoch,
                modules={
                    'model_G': self.model_G, 'model_D': self.model_D, 
                    'optimizer_G': self.optimizer_G, 'optimizer_D': self.optimizer_D, 
                    'scheduler_G': self.scheduler_G, 'scheduler_D': self.scheduler_D},
                metric=None
            )
            logger.info('Model saved')
