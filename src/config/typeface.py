import tarfile
from pathlib import Path

import torch
from loguru import logger
from src.disk import disk
from src.utils.download import download_dataset
from src.logger.simple import Logger
from src.metrics.accuracy import TopKAccuracy
from src.storage.simple import Storage
from src.training.img_classifier import ImgClassifierTrainer
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models.vgg import vgg11
from torchvision import transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Config:
    def __init__(self):
        disk.login()
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Using device: {device}')

        download_dataset('Typefaces')

        model = vgg11(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(4096, 2500)
        model = model.to(device)
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)
        storage = Storage('checkpoints/typeface')
        metric_logger = Logger()

        transform = T.Compose([
            T.ToTensor(),
            T.Resize((64, 196)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = ImageFolder('data/Typefaces/train', transform)
        val_dataset = ImageFolder('data/Typefaces/val', transform)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)

        self.trainer = ImgClassifierTrainer(
            model=model,
            criterion=criterion,
            metric=TopKAccuracy((1, 5, 10)),
            optimizer=optimizer,
            scheduler=scheduler,
            storage=storage,
            logger=metric_logger,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            max_epoch=3,
            device=device
        )

    def run(self):
        self.trainer.run()
