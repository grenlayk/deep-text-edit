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
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(range(0, total_epochs, 5)),
            gamma=0.2,
            verbose=True
        )

        batch_size = 56
        train_dataset = CustomDataset(data_path / 'train', training=True)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8
        )

        val_dataset = CustomDataset(data_path / 'val')
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8
        )

        test_dataset = CustomDataset(data_path / 'test')
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8
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
