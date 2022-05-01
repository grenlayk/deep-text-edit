from pathlib import Path

import zipfile
import torch
from fastai.vision.models import resnet18
from fastai.vision.models.unet import DynamicUnet
from loguru import logger
from src.data.color import CustomDataset
from src.disk import disk
from src.logger.simple import Logger
from src.models.rrdb import RRDBNet
from src.storage.simple import Storage
from src.training.gan_colorization import GANColorizationTrainer
from src.utils.warmup import WarmupScheduler
from src.models.nlayer_discriminator import NLayerDiscriminator
from src.losses import Compose, VGGPerceptualLoss

class Identity(torch.nn.Module):
    def forward(self, x):
        return x

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
        m = resnet18(True)
        m = torch.nn.Sequential(*list(m.children())[:-2])
        model_G = DynamicUnet(m, 3, (128, 128)).to(device) 
        model_D = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=(lambda x : Identity())).to(device)
        criterion = Compose(
            [torch.nn.L1Loss().to(device), VGGPerceptualLoss().to(device)],
            [1, 0.1],
        ).to(device)
        criterion_gan = torch.nn.MSELoss(reduction='mean').to(device)
        lambda_gan = 0.3

        batch_size = 16
        train_dataset = CustomDataset(data_path / 'train', crop_size=32)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        logger.info(f'Train size: {len(train_dataloader)} x {batch_size}')

        val_dataset = CustomDataset(data_path / 'val', cut=(200 / 5000))
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False
        )
        logger.info(f'Validate size: {len(val_dataloader)} x {1}')

        optimizer_G = torch.optim.Adam(model_G.parameters(), lr=1e-4)
        optimizer_D = torch.optim.Adam(model_D.parameters(), lr=2e-4)

        scheduler_G = WarmupScheduler(
            optimizer=optimizer_G,
            warmup_epochs=len(train_dataloader) // 3,
            scheduler=torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer_G,
                gamma=0.9**(1 / len(train_dataloader)),
            )
        )

        metric_logger = Logger(print_freq=100, image_freq=100, project_name='gan_colorization')
        storage = Storage('./checkpoints/gan_colorization')

        self.trainer = GANColorizationTrainer(
            model_G,
            model_D,
            criterion,
            criterion_gan,
            lambda_gan,
            optimizer_G,
            optimizer_D,
            scheduler_G,
            train_dataloader,
            val_dataloader,
            total_epochs,
            metric_logger,
            storage
        )

    def run(self):
        self.trainer.run()
