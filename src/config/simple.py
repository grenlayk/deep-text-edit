from training.simple import SimpleTrainer
from src.models.simple import Model
from src.models.simple import Loss
from src.logger.storage import Storage
from src.logger.simple import Logger
from src.data.simple import SimpleDataset
from torch.utils.data import DataLoader

class Config:
    def __init__(self):
        model = Model()
        criterion = Loss()
        optimizer = None
        storage = Storage()
        logger = Logger()
        train_dataloader = DataLoader(SimpleDataset('local/train'))
        val_dataloader = DataLoader(SimpleDataset('local/val'))
        
        self.trainer = SimpleTrainer(model, criterion, optimizer, storage, 
                                     logger, train_dataloader, val_dataloader)

    def run():
        self.trainer.run()
