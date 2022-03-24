
import torch
from loguru import logger
from disk import disk
from pathlib import Path
from src.logger import Logger
from src.data.baseline import  download_data, setup_data
from src.models.rrdb import RRDB_pretrained
from src.training.baseline import Trainer
from src.storage.simple import Storage
from torch.utils.data import DataLoader

class Config:
    def __init__(self):
        disk.login()

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Using device: {device}')
        data_dir = Path("data/imgur5k")
        if not data_dir.exists():
            data_dir.mkdir()
            style_dir = data_dir / 'style'
            content_dir = data_dir / 'content'
            style_dir.mkdir()
            content_dir.mkdir()
            download_data(Path("data/IMGUR5k.tar.lz4"), style_dir)
    
        train_dataloader = DataLoader(setup_data(style_dir / 'train', content_dir / 'train'))
        val_dataloader = DataLoader(setup_data(style_dir / 'val', content_dir / 'val'))

        total_epochs = 1 #20
        model = RRDB_pretrained().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(range(0, total_epochs, 5)),
            gamma=0.2
        )

        storage = Storage('checkpoints/baseline')

        logger = Logger()

        self.trainer = Trainer(
            model,
            optimizer,
            scheduler,
            train_dataloader,
            val_dataloader,
            storage,
            logger,
            total_epochs,
            device=device
        )

    def run(self):
        self.trainer.run()
