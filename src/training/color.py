import time
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
        # Assumption is that im_lab is of size [1,3,224,224]
        # print(im_lab.size(),img_ab.size())
        np_img = im_lab[0].cpu().detach().numpy().transpose(1, 2, 0)
        lab = np.empty([*np_img.shape[0:2], 3], dtype=np.float32)
        lab[:, :, 0] = np.squeeze(((np_img + 1) * 50))
        lab[:, :, 1:] = img_ab[0].cpu().detach().numpy().transpose(1, 2, 0) * 127
        np_img = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
        color_im = torch.stack([torchvision.transforms.ToTensor()(np_img)], dim=0)
        return color_im

    def run(self):
        for epoch in range(self.total_epochs):
            # Training step
            self.model.train()

            # x: l_img x3
            # y: ab_img
            for idx, (l_img, ab_img) in enumerate(self.train_dataloader):
                # Skip bad data
                if not l_img.ndim:
                    continue

                l_img = l_img.to(self.device)
                ab_img = ab_img.to(self.device)

                # Initialize Optimizer
                self.optimizer.zero_grad()

                # Forward Propagation
                output_ab = self.model(l_img)

                # Back propogation
                loss = self.criterion(output_ab, ab_img.float())
                loss.backward()

                # Weight Update
                self.optimizer.step()

                # Reduce Learning Rate
                self.scheduler.step()

                # Print stats after every point_batches
                self.logger.log_train(losses={'main': loss.item()}, images={'input': l_img})


            # Validation Step
            # Intialize Model to Eval Mode for validation
            self.model.eval()
            for idx, (l_img, ab_img) in enumerate(self.val_dataloader):
                # Skip bad data
                if not l_img.ndim:
                    continue

                l_img = l_img.to(self.device)
                ab_img = ab_img.to(self.device)

                # Forward Propagation
                output_ab = self.model(l_img)

                # Loss Calculation
                loss = self.criterion(output_ab, ab_img.float())

                self.logger.log_val(losses={'main': loss.item()}, images={'input': l_img})
                
            self.logger.end_val()

            # Save the Model to disk
            self.storage.save(epoch, {'model': self.model, 'optimizer': self.optimizer, 'scheduler': self.scheduler}, None)
            logger.info(f'Model saved')

        # ### Inference

        # Inference Step
        logger.info(f'-------------- Test dataset validation --------------')

        for idx, (l_img, ab_img) in enumerate(self.test_dataloader):
            # Skip bad data
            if not l_img.ndim:
                continue

            l_img = l_img.to(self.device)
            ab_img = ab_img.to(self.device)

            # Intialize Model to Eval Mode
            self.model.eval()

            # Forward Propagation
            output_ab = self.model(l_img)

            # resize l_img
            l_img = torchvision.transforms.Resize(224)(l_img)

            # Adding l channel to ab channels
            color_img = self.concatenate_and_colorize(torch.stack([l_img[:, 0, :, :]], dim=1), output_ab)

            save_path = Path('outputs')
            save_path.mkdir(exist_ok=True)
            save_path /= f'img{idx}.jpg'
            save_image(color_img[0], save_path)

            # Loss Calculation
            loss = self.criterion(output_ab, ab_img.float())

            self.logger.log_val(losses={'main': loss.item()}, images={'input': l_img})
                
        self.logger.end_val()
