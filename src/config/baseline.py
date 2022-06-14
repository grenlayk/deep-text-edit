import torch
from loguru import logger as info_logger
from src.disk import disk
from pathlib import Path
from src.logger.simple import Logger
from src.data.baseline import BaselineDataset
from src.utils.download import download_dataset
from src.models.rrdb import RRDB_pretrained
from src.training.baseline import Trainer
from src.storage.simple import Storage
from src.losses.perceptual import VGGPerceptualLoss
from src.losses.ocr import OCRLoss
from torch.utils.data import DataLoader


class Config:
    def __init__(self):
        disk.login()

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        info_logger.info(f'Using device: {device}')

        style_dir = Path('data/IMGUR5K')
        download_dataset('IMGUR5K')
        batch_size = 4
        train_dataloader = DataLoader(BaselineDataset(style_dir / 'train'), shuffle=True, batch_size=batch_size)
        val_dataloader = DataLoader(BaselineDataset(style_dir / 'val'), batch_size=batch_size)

        total_epochs = 20
        model = RRDB_pretrained().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(range(0, total_epochs, 5)),
            gamma=0.2
        )

        ocr_coef = 0.5
        perceptual_coef = 0.5

        storage = Storage('checkpoints/baseline')

        logger = Logger(image_freq=100, project_name='Baseline')

        self.trainer = Trainer(
            model,
            optimizer,
            scheduler,
            train_dataloader,
            val_dataloader,
            storage,
            logger,
            total_epochs,
            device,
            ocr_coef,
            perceptual_coef,
            OCRLoss(),
            VGGPerceptualLoss(),
        )

    def run(self):
        self.trainer.run()


config = Config().run()
