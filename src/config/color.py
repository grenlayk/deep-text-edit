import zipfile
from pathlib import Path

import torch
from fastai.vision.models import resnet18
from fastai.vision.models.unet import DynamicUnet
from loguru import logger
from src.data.color import ColorDataset
from src.disk import disk
from src.logger.simple import Logger
from src.losses import ComposeLoss, VGGPerceptualLoss
from src.storage.simple import Storage
from src.training.color import ColorizationTrainer
from src.utils.warmup import WarmupScheduler
from torch import nn


class Config:
    def __init__(self):
        disk.login()
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Using device: {device}')

        data_path = Path('data/Coco')

        if not data_path.exists():
            disk.download(data_path.with_suffix('.zip'), data_path.with_suffix('.zip'))
            with zipfile.ZipFile(data_path.with_suffix('.zip'), 'r') as zip_ref:
                zip_ref.extractall('data/')

        total_epochs = 20

        resnet = resnet18(True)
        resnet_blocks = nn.Sequential(*list(resnet.children())[:-2])
        model = DynamicUnet(resnet_blocks, 3, (128, 128)).to(device)
        criterion = ComposeLoss(
            [torch.nn.L1Loss().to(device), VGGPerceptualLoss().to(device)],
            [1, 0.125],
        ).to(device)

        batch_size = 16
        train_dataset = ColorDataset(data_path / 'train', crop_size=64)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        logger.info(f'Train size: {len(train_dataloader)} x {batch_size}')

        val_dataset = ColorDataset(data_path / 'val', cut=(200 / 5000))
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False
        )
        logger.info(f'Validate size: {len(val_dataloader)} x {1}')

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = WarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=len(train_dataloader) // 3,
            scheduler=torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer,
                gamma=0.9**(1 / len(train_dataloader)),
            )
        )

        metric_logger = Logger(print_freq=100, image_freq=100, project_name='colorization_unet')
        storage = Storage('./checkpoints/colorization_unet')

        self.trainer = ColorizationTrainer(
            model,
            criterion,
            optimizer,
            scheduler,
            train_dataloader,
            val_dataloader,
            None,
            total_epochs,
            metric_logger,
            storage
        )

    def run(self):
        self.trainer.run()
