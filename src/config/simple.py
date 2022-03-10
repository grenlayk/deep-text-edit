from src.data.simple import SimpleDataset
from src.logger.simple import Logger
from src.losses.simple import Loss
from src.models.simple import Model
from src.storage.simple import Storage
from src.training.simple import SimpleTrainer
from torch.utils.data import DataLoader


class Config:
    def __init__(self):
        model = Model()
        criterion = Loss()
        optimizer = None
        storage = Storage('path/to/storage')
        logger = Logger()
        train_dataloader = DataLoader(SimpleDataset('local/train'))
        val_dataloader = DataLoader(SimpleDataset('local/val'))

        self.trainer = SimpleTrainer(model, criterion, optimizer, storage,
                                     logger, train_dataloader, val_dataloader)

    def run(self):
        self.trainer.run()
