import torch
from loguru import logger as info_logger
from src.disk import disk
from pathlib import Path
from src.logger.simple import Logger
from src.data.baseline import BaselineDataset
from src.utils.download import download_dataset
from src.models.embedders import ContentResnet, StyleResnet
from src.models.stylegan import StyleBased_Generator
from src.training.autoencoder import AutoencoderTrainer
from src.storage.simple import Storage
from torch.utils.data import DataLoader


class Config:
    def __init__(self):
        disk.login()

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        info_logger.info(f'Using device: {device}')
        style_dir = Path('data/IMGUR5K')
        download_dataset('IMGUR5K')
        batch_size = 16
        train_dataloader = DataLoader(BaselineDataset(style_dir / 'train'), shuffle=True, batch_size=batch_size)
        val_dataloader = DataLoader(BaselineDataset(style_dir / 'val'), batch_size=batch_size)

        total_epochs = 500

        model = StyleBased_Generator(dim_latent=512)
        model.to(device)

        style_embedder = StyleResnet().to(device) 
        content_embedder = ContentResnet().to(device)

        optimizer = torch.optim.AdamW(
            list(model.parameters()) +
            list(style_embedder.parameters()) +
            list(content_embedder.parameters()),
            lr=1e-3,
            weight_decay=1e-6
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.8
        )

        storage = Storage('checkpoints/stylegan(pretrained_on_content)')

        logger = Logger(
            image_freq=100,
            project_name='Autoencoder',
            config={
                'img_size': (192, 64)
            }
        )

        self.trainer = AutoencoderTrainer(
            model,
            style_embedder,
            content_embedder,
            optimizer,
            scheduler,
            train_dataloader,
            val_dataloader,
            storage,
            logger,
            total_epochs,
            device,
            torch.nn.L1Loss()
        )

    def run(self):
        self.trainer.run()
