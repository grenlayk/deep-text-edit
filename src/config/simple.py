from src.data.simple import SimpleDataset
from src.logger.simple import Logger
from src.storage.simple import Storage
from src.training.simple import SimpleTrainer
from torch.utils.data import DataLoader
import torch


class Config:
    def __init__(self):
        model = None
        criterion = None
        optimizer = None
        storage = Storage('path/to/storage')
        logger = Logger('./checkpoints/simple', project_name='simple')
        train_dataloader = DataLoader(SimpleDataset('local/train'))
        val_dataloader = DataLoader(SimpleDataset('local/val'))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        max_epoch = 20

        self.trainer = SimpleTrainer(model, criterion, None, optimizer, None, train_dataloader, val_dataloader,
                                     storage, logger, max_epoch, device)

    def run(self):
        self.trainer.run()
