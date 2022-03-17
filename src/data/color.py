from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from loguru import logger
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.files = list(root_dir.iterdir())
        logger.info(f'File[0]: {self.files[0]}, Total Files: {len(self.files)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        try:
            rgb_img = cv2.imread(str(self.files[index]))
            if rgb_img is None:
                raise Exception
            rgb_img = rgb_img.astype(np.float32)
            rgb_img /= 255.0

            rgb_res_img = cv2.resize(rgb_img, (300, 300))
            l_img = cv2.cvtColor(rgb_res_img, cv2.COLOR_BGR2Lab)[:, :, 0]

            l_img = l_img / 50.0 - 1.0
            l_img = torchvision.transforms.ToTensor()(l_img)
            l_img = l_img.expand(3, -1, -1)

            rgb_enc_img = cv2.resize(rgb_img, (224, 224))
            lab_encoder_img = cv2.cvtColor(rgb_enc_img, cv2.COLOR_BGR2Lab)

            a_encoder_img = lab_encoder_img[:, :, 1] / 128.0
            b_encoder_img = lab_encoder_img[:, :, 2] / 128.0
            a_encoder_img = torch.stack([torch.Tensor(a_encoder_img)])
            b_encoder_img = torch.stack([torch.Tensor(b_encoder_img)])
            ab_encoder_img = torch.cat([a_encoder_img, b_encoder_img], dim=0)

            return l_img, ab_encoder_img

        except Exception as e:
            logger.error(f'Exception at {self.files[index]}, {e}')
            return torch.tensor(-1), torch.tensor(-1)
        
