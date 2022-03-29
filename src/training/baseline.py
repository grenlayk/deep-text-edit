import torch
from src.logger.simple import Logger
from src.storage.simple import Storage
from src.losses import ocr, perceptual
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
                 coef_perceptual_loss: float):
        
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.total_epochs = total_epochs
        self.logger = logger
        self.storage = storage
        self.ocr_loss = ocr.OCRLoss()
        self.perceptual_loss = perceptual.VGGPerceptualLoss()
        self.coef_ocr = coef_ocr_loss
        self.coef_perceptual = coef_perceptual_loss

    # def normalize(self, batch):
    #     mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    #     std =  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    #     return (batch - mean) / std

    def concat_batches(self, batch_1, batch_2):
        '''
        Concatenate 2 * (Bx3xHxW) image batches along channels axis -> (Bx2*3xHxW)
        '''
        return torch.cat((batch_1, batch_2), 1)

    
    def train(self):
        logger.info('Start training')
        #self.setup_data_one_image()
        self.model.train()

        for style_batch, content_batch, label_batch in self.train_dataloader:
            concat_batches = (self.concat_batches(style_batch, content_batch)).to(self.device)

            self.optimizer.zero_grad()

            res = self.model(concat_batches).cpu()
            ocr_loss = self.ocr_loss(res, label_batch)
            perceptual_loss = self.perceptual_loss(style_batch.cpu(), res)
            loss = self.coef * ocr_loss +  self.coef_perceptual * perceptual_loss
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.logger.log_train(
                losses={'ocr_loss': ocr_loss.item(), 'perceptual_loss': perceptual_loss.item(), 'full_loss': loss.item()},
                images={'style': style_batch, 'content': content_batch, 'result': res}
            )

            # res = res.clamp_(0, 1).squeeze(0).detach()
            # output = res.permute(1,2,0).numpy()
            # output = (output * 255.0).round()
            # cv2.imwrite('1.jpg', output)


    def validate(self, epoch):
        self.model.eval()

        for style_batch, content_batch, label_batch in self.val_dataloader:
            concat_batches = (self.concat_batches(style_batch, content_batch)).to(self.device)
            res = (self.model(concat_batches).cpu())
            ocr_loss = self.ocr_loss(res, label_batch)
            perceptual_loss = self.perceptual_loss(style_batch.cpu(), res)
            loss = self.coef_ocr * ocr_loss +  perceptual_loss
            
            self.logger.log_val(
                losses={'ocr_loss': ocr_loss.item(), 'perceptual_loss': perceptual_loss.item(), 'full_loss': loss.item()},
                images={'style': style_batch, 'content': content_batch, 'result': res}
            )

        avg_losses, _ = self.logger.end_val()
        self.storage.save(
            epoch,
            {'model': self.model, 'optimizer': self.optimizer, 'scheduler': self.scheduler},
            avg_losses['metric']
        )


    def run(self):
        for epoch in range(self.total_epochs):
            self.train()
            with torch.no_grad():
                self.validate(epoch)