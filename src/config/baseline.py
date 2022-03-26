
import torch
import sys
from loguru import logger as info_logger
origin_path = sys.path
sys.path.append("..")
sys.path = origin_path
from src.disk import disk
from pathlib import Path
from src.logger.simple import Logger
from src.data.baseline import  download_data, setup_dataset
from src.models.rrdb import RRDB_pretrained
from src.training.baseline import Trainer
from src.storage.simple import Storage
from torch.utils.data import DataLoader

class Config:
    def __init__(self):
        disk.login()

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        info_logger.info(f'Using device: {device}')
        data_dir = Path("data")
        style_dir = data_dir / 'IMGUR5K_small'
        content_dir = data_dir / 'content'
        if not data_dir.exists():
            data_dir.mkdir()
            download_data(Path("data/IMGUR5K_small.tar"), data_dir)
    
        train_dataloader = DataLoader(setup_dataset(style_dir / 'train', content_dir / 'train'))
        val_dataloader = DataLoader(setup_dataset(style_dir / 'val', content_dir / 'val'))

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


config = Config().run()