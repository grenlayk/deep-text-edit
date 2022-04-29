import time
import torch
from loguru import logger
from torchvision.utils import save_image
from pathlib import Path


class ColorizationTrainer:
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 scheduler,
                 train_dataloader,
                 val_dataloader,
                 test_dataloader,
                 total_epochs,
                 logger,
                 storage):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.val_dataloader = val_dataloader
        self.total_epochs = total_epochs
        self.logger = logger
        self.storage = storage

    def train(self):
        self.model.train()

        # inputs: bw_img x3
        # target: rgb_img
        for inputs, targets in self.train_dataloader:

            # Skip bad data
            if not inputs.ndim:
                continue

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Initialize Optimizer
            self.optimizer.zero_grad()

            # Forward Propagation
            preds = self.model(inputs)

            # Back propogation
            loss = self.criterion(preds, targets)
            loss['total'].backward()

            # Weight Update
            self.optimizer.step()
            # Reduce Learning Rate
            self.scheduler.step()

            # Print stats after every point_batches
            self.logger.log_train(
                losses={**loss, 'lr': self.scheduler.get_last_lr()[0]},
                images={'input': inputs, 'pred': torch.FloatTensor(preds.cpu().detach()), 'target': targets},
            )

    def validate(self, epoch: int):
        # Validation Step
        # Intialize Model to Eval Mode for validation
        self.model.eval()
        for inputs, targets in self.val_dataloader:
            # Skip bad data
            if not inputs.ndim:
                continue

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward Propagation
            preds = self.model(inputs)

            # Loss Calculation
            loss = self.criterion(preds, targets)

            self.logger.log_val(
                losses=loss,
                images={'input': inputs, 'pred': torch.FloatTensor(preds.cpu().detach()), 'target': targets},
            )

        return self.logger.end_val()

    def run(self):
        for epoch in range(self.total_epochs):
            start = time.time_ns()
            self.train()
            torch.cuda.empty_cache()
            logger.success(f'Finished training epoch #{epoch} in {(time.time_ns() - start) / 1e9}s')
            with torch.no_grad():
                _, _ = self.validate(epoch)

            # Save the Model to disk
            self.storage.save(
                epoch,
                {'model': self.model, 'optimizer': self.optimizer, 'scheduler': self.scheduler},
                None
            )
            logger.info('Model saved')
            torch.cuda.empty_cache()
