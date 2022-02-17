import time

import cv2
import numpy as np
import torch
import torchvision
from loguru import logger
from torchvision.models.resnet import resnet50
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
                 batch_size,
                 point_batches,
                 model_save_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.resnet = resnet50(pretrained=True, progress=True).to(device).eval()
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.point_batches = point_batches
        self.model_save_path = model_save_path

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
        for epoch in range(20):
            logger.info(f'Starting epoch #{epoch + 1}')

            # *** Training step ***
            loop_start = time.time()
            avg_loss = 0.0
            batch_loss = 0.0
            main_start = time.time()
            self.model.train()

            for idx, (img_l_encoder, img_ab_encoder, img_l_resnet, img_rgb,
                      file_name) in enumerate(self.train_dataloader):
                # *** Skip bad data ***
                if not img_l_encoder.ndim:
                    continue

                # *** Move data to GPU if available ***
                img_l_encoder = img_l_encoder.cuda()
                img_ab_encoder = img_ab_encoder.cuda()
                img_l_resnet = img_l_resnet.cuda()

                # *** Initialize Optimizer ***
                self.optimizer.zero_grad()

                # *** Forward Propagation ***
                img_embs = self.resnet(img_l_resnet.float())
                output_ab = self.model(img_l_encoder, img_embs)

                # *** Back propogation ***
                loss = self.criterion(output_ab, img_ab_encoder.float())
                loss.backward()

                # *** Weight Update ****
                self.optimizer.step()

                # *** Reduce Learning Rate ***
                self.scheduler.step()

                # *** Loss Calculation ***
                avg_loss += loss.item()
                batch_loss += loss.item()

                # *** Print stats after every point_batches ***
                if (idx + 1) % self.point_batches == 0:
                    loop_end = time.time()
                    logger.info(
                        f'Batch: {idx + 1}, '
                        f'Batch time: {loop_end - loop_start: 3.1f}s, '
                        f'Batch Loss: {batch_loss / self.point_batches:7.5f}'
                    )
                    loop_start = time.time()
                    batch_loss = 0.0

            # *** Print Training Data Stats ***
            train_loss = avg_loss / len(self.train_dataloader) * self.batch_size
            logger.info(f'Training Loss: {train_loss}, Processed in {time.time() - main_start:.3f}s')

            # *** Validation Step ***
            avg_loss = 0.0
            loop_start = time.time()
            # *** Intialize Model to Eval Mode for validation ***
            self.model.eval()
            for idx, (img_l_encoder, img_ab_encoder, img_l_resnet, img_rgb,
                      file_name) in enumerate(self.val_dataloader):
                # *** Skip bad data ***
                if not img_l_encoder.ndim:
                    continue

                # *** Move data to GPU if available ***
                img_l_encoder = img_l_encoder.cuda()
                img_ab_encoder = img_ab_encoder.cuda()
                img_l_resnet = img_l_resnet.cuda()

                # *** Forward Propagation ***
                img_embs = self.resnet(img_l_resnet.float())
                output_ab = self.model(img_l_encoder, img_embs)

                # *** Loss Calculation ***
                loss = self.criterion(output_ab, img_ab_encoder.float())
                avg_loss += loss.item()

            val_loss = avg_loss / len(self.val_dataloader) * self.batch_size
            logger.info(f'Validation Loss: {val_loss}, Processed in {time.time() - loop_start:.3f}s')

            logger.success(
                f'Finished epoch #{epoch + 1} out of {self.total_epochs}. '
                f'Train loss: {train_loss:.5f}, val loss: {val_loss:.5f}'
            )
            # if config.wb_enabled:
            #    wandb.log({"Validation loss": val_loss, "Train loss": train_loss}, step=epoch)

            # *** Save the Model to disk ***
            checkpoint = {'model_state_dict': self.model.state_dict(),
                          'optimizer_state_dict': self.optimizer.state_dict(),
                          'scheduler_state_dict': self.scheduler.state_dict(),
                          'train_loss': train_loss, 'val_loss': val_loss}
            torch.save(checkpoint, self.model_save_path / str(epoch + 1))
            logger.info(f'Model saved at: {self.model_save_path / str(epoch + 1)}')

        # ### Inference

        # logger.info(f'Test: {len(test_dataloader)} Total Image: {len(test_dataloader)}')

        # *** Inference Step ***
        avg_loss = 0.0
        loop_start = time.time()
        batch_start = time.time()
        batch_loss = 0.0

        for idx, (img_l_encoder, img_ab_encoder, img_l_resnet, img_rgb, file_name) in enumerate(self.test_dataloader):
            # *** Skip bad data ***
            if not img_l_encoder.ndim:
                continue

            # *** Move data to GPU if available ***
            img_l_encoder = img_l_encoder.cuda()
            img_ab_encoder = img_ab_encoder.cuda()
            img_l_resnet = img_l_resnet.cuda()

            # *** Intialize Model to Eval Mode ***
            self.model.eval()

            # *** Forward Propagation ***
            img_embs = self.resnet(img_l_resnet.float())
            output_ab = self.model(img_l_encoder, img_embs)

            # *** Adding l channel to ab channels ***
            color_img = self.concatente_and_colorize(torch.stack([img_l_encoder[:, 0, :, :]], dim=1), output_ab)

            save_path = Path('outputs')
            save_path.mkdir(exist_ok=True)
            save_path /= file_name[0]
            save_image(color_img[0], save_path)

            # *** Printing to Tensor Board ***
            # grid = torchvision.utils.make_grid(color_img)
            # writer.add_image('Output Lab Images', grid, 0)

            # *** Loss Calculation ***
            loss = self.criterion(output_ab, img_ab_encoder.float())
            avg_loss += loss.item()
            batch_loss += loss.item()

            if (idx + 1) % self.point_batches == 0:
                batch_end = time.time()
                logger.info(
                    f'Batch: {idx + 1}, '
                    f'Processing time for {self.point_batches}: {batch_end - batch_start:.3f}s, '
                    f'Batch Loss: {batch_loss / self.point_batches}'
                )
                batch_start = time.time()
                batch_loss = 0.0

        test_loss = avg_loss / len(self.test_dataloader)
        logger.info(f'Test Loss: {test_loss} Processed in {time.time() - loop_start:.3f}s')
        pass
