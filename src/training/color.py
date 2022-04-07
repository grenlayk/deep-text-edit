from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from loguru import logger
from torchvision.utils import save_image


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

    def concatenate_and_colorize(self, im_lab, img_ab):
        # print(im_lab.size(),img_ab.size())
        im_lab = torchvision.transforms.Resize(224)(im_lab)
        np_img = im_lab[0].cpu().detach().numpy().transpose(1, 2, 0)
        lab = np.empty([*np_img.shape[0:2], 3], dtype=np.float32)
        lab[:, :, 0] = np.squeeze(((np_img + 1) * 50))
        lab[:, :, 1:] = img_ab[0].cpu().detach().numpy().transpose(1, 2, 0) * 127
        np_img = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
        color_im = torch.stack([torchvision.transforms.ToTensor()(np_img)], dim=0)
        return color_im

    def train(self):
        self.model.train()

        # inputs: l_img x3
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
            loss.backward()

            # Weight Update
            self.optimizer.step()

            # Print stats after every point_batches
            self.logger.log_train(
                losses={'main': loss.item()},
                images={'input': inputs, 'pred': preds, 'target': targets},
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
                losses={'main': loss.item()},
                images={'input': inputs, 'pred': preds, 'target': targets},
            )

        return self.logger.end_val()

    def run(self):
        for epoch in range(self.total_epochs):
            self.train()
            with torch.no_grad():
                avg_losses, _ = self.validate(epoch)
                # Reduce Learning Rate
                self.scheduler.step(avg_losses['main'])

            # Save the Model to disk
            self.storage.save(
                epoch,
                {'model': self.model, 'optimizer': self.optimizer, 'scheduler': self.scheduler},
                None
            )
            logger.info('Model saved')

        # Inference Step
        logger.info('-------------- Test dataset validation --------------')

        for idx, (inputs, targets) in enumerate(self.test_dataloader):
            # Skip bad data
            if not inputs.ndim:
                continue

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Intialize Model to Eval Mode
            self.model.eval()

            # Forward Propagation
            preds = self.model(inputs)

            save_path = Path('outputs')
            save_path.mkdir(exist_ok=True)
            save_path /= f'img{idx}.jpg'
            save_image(preds[0], save_path)

            # Loss Calculation
            loss = self.criterion(preds, targets)

            self.logger.log_val(losses={'main': loss.item()}, images={'input': inputs})

        self.logger.end_val()
