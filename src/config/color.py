from pathlib import Path

import zipfile
import torch
from loguru import logger
from src.data.color import CustomDataset
from src.disk import disk
from src.logger.simple import Logger
from src.models.rrdb import RRDBNet
from src.storage.simple import Storage
from src.training.color import ColorizationTrainer
from src.utils.warmup import WarmupScheduler


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

        total_epochs = 20  # 20
        model = RRDBNet(3, 3, 64, 10, gc=32).to(device)
        criterion = torch.nn.MSELoss(reduction='mean').to(device)

        batch_size = 32
        train_dataset = CustomDataset(data_path / 'train', crop_size=32)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8
        )
        logger.info(f'Train size: {len(train_dataloader)} x {batch_size}')

        val_dataset = CustomDataset(data_path / 'val')
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8
        )
        logger.info(f'Validate size: {len(val_dataloader)} x {1}')

        test_dataset = CustomDataset(data_path / 'test')
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8
        )
        logger.info(f'Test size: {len(test_dataloader)} x {1}')

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        scheduler = WarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=2 * len(train_dataloader),
            scheduler=torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer,
                gamma=0.9**(1 / len(train_dataloader)),
            )
        )

        metric_logger = Logger(print_freq=100, image_freq=100, project_name='colorization_rrdb')
        storage = Storage('./checkpoints/colorization_rrdb')

        self.trainer = ColorizationTrainer(
            model,
            criterion,
            optimizer,
            scheduler,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            total_epochs,
            metric_logger,
            storage
        )

    def run(self):
        self.trainer.run()
