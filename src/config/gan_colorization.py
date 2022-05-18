from pathlib import Path

import zipfile
import torch
from fastai.vision.models import resnet18
from fastai.vision.models.unet import DynamicUnet
from loguru import logger
from src.data.color import ColorDataset
from src.disk import disk
from src.logger.simple import Logger
from src.storage.simple import Storage
from src.training.gan_colorization import GANColorizationTrainer
from src.utils.warmup import WarmupScheduler
from src.models.nlayer_discriminator import NLayerDiscriminator
from src.losses import ComposeLoss, VGGPerceptualLoss

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

        m = torch.nn.Sequential(*list(resnet18(True).children())[:-2])
        model_G = DynamicUnet(m, 3, (128, 128)).to(device) 
        model_D = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=(lambda x : Identity())).to(device)

        lambda_l1 = 2
        lambda_vgg = 0.125
        lambda_gan = 0.15

        criterion = ComposeLoss(
            losses=[torch.nn.L1Loss().to(device), VGGPerceptualLoss().to(device)],
            coefs=[lambda_l1, lambda_vgg],
        ).to(device)
        criterion_gan = torch.nn.MSELoss(reduction='mean').to(device)
        
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

        optimizer_G = torch.optim.Adam(model_G.parameters(), lr=1e-4)
        optimizer_D = torch.optim.Adam(model_D.parameters(), lr=5e-4)

        scheduler_G = WarmupScheduler(
            optimizer=optimizer_G,
            warmup_epochs=len(train_dataloader) // 3,
            scheduler=torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer_G,
                gamma=0.9**(1 / len(train_dataloader)),
            )
        )
        scheduler_D = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer_D,
            gamma=0.9**(1 / len(train_dataloader)),
        )

        project_name = 'gan_colorization'
        metric_logger = Logger(print_freq=100, image_freq=100, project_name=project_name)
        storage = Storage(f'./checkpoints/{project_name}')

        self.trainer = GANColorizationTrainer(
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
            metric_logger,
            storage
        )

    def run(self):
        self.trainer.run()
