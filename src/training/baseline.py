import sys
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
from pathlib import Path
import numpy as np
origin_path = sys.path
sys.path.append("..")
sys.path = origin_path
from disk import  disk
from losses import ocr, perceptual
from torch.utils.data import DataLoader

from data.baseline import BaselineDataset


class Trainer:
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 train_dataloader,
                 val_dataloader,
                 test_dataloader,
                 total_epochs,
                 batch_size,
                 point_batches,
                 model_save_path):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.point_batches = point_batches
        self.model_save_path = model_save_path
        self.ocr_loss = ocr.OCRLoss('/home/nikita/Huawei Project/text-deep-fake/src/models/crnn.pth')
        self.perceptual_loss = perceptual.VGGPerceptualLoss()

    def normalize(self, batch):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std =  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (batch - mean) / std

    def concat_batches(self, batch_1, batch_2):
        '''
        Concatenate 2 * (Bx3xHxW) image batches along channels axis -> (Bx2*3xHxW)
        '''
        return torch.cat((batch_1, batch_2), 3)

    def setup_data_one_image(self):
        # disk.login()
        # disk.download('/data/cropped/train/01dWS6L_2.png', '../../data/raw/imgur5k/train/style/1.png')
        
        img = Image.new('RGB', (256, 64), color = (255, 255, 255))
        fnt = ImageFont.truetype('../utils/VerilySerifMono.otf', 40)
        d = ImageDraw.Draw(img)
        d.text((60, 10), "PEACE", font=fnt, fill=(0, 0, 0))
        img.save('../../data/raw/imgur5k/train/content/peace.png')

        self.train_dataloader = DataLoader(BaselineDataset(Path('../../data/raw/imgur5k/train')))
    
    def train(self):
        self.setup_data_one_image()
        for style_batch, content_batch, label_batch in self.train_dataloader:
            print(style_batch.shape, content_batch.shape)
            self.model.train()
            concat_batches = (self.concat_batches(style_batch, content_batch))

            #norm_bathces = self.normalize(concat_batches)
            res = self.model(concat_batches)
            loss = self.ocr_loss(res, label_batch) + self.perceptual_loss(style_batch, res)
            loss.backward()
            self.optimizer.step()
            res = res.clamp_(0, 1).squeeze(0).detach()
            output = res.permute(1,2,0).numpy()
            output = (output * 255.0).round()
            cv2.imwrite('1.jpg', output)

