import torch
from loguru import logger as info_logger
from src.disk import disk
from pathlib import Path
from src.logger.simple import Logger
from src.data.baseline import  BaselineDataset
from src.utils.download import download_data, unarchieve 
from src.models.stylegan import StyleBased_Generator
from src.training.stylegan import Trainer
from src.storage.simple import Storage
from src.losses.perceptual import VGGPerceptualLoss
from src.losses.gram import VGGGramLoss
from torch.utils.data import DataLoader

class Config:
    def __init__(self):
        disk.login()

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        info_logger.info(f'Using device: {device}')

        data_dir = Path("data/imgur")
        style_dir = data_dir / 'IMGUR5K_small'
        if not data_dir.exists():
            data_dir.mkdir()
            local_path = download_data(Path("data/IMGUR5K_small.tar"), data_dir)
            unarchieve(local_path)
        
        batch_size = 16
        train_dataloader = DataLoader(BaselineDataset(style_dir / 'train'), shuffle=True, batch_size = batch_size)
        val_dataloader = DataLoader(BaselineDataset(style_dir / 'val'), batch_size = batch_size)

        total_epochs = 500
        model = StyleBased_Generator(dim_latent=512).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(range(0, total_epochs, 20)),
            gamma=0.2
        )

        content_coef = 0.2
        style_coef = 0.8
        style_loss = VGGPerceptualLoss()
        content_loss = VGGGramLoss()

        project_name = 'stylegan_olya_tests'
        
        storage = Storage(f'checkpoints/{project_name}')
        logger = Logger(image_freq=100, project_name=project_name)

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
            content_coef,
            style_coef,
            content_loss,
            style_loss
        )

    def run(self):
        self.trainer.run()
