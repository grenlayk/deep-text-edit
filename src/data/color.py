from pathlib import Path
from random import shuffle
from typing import Optional, Tuple, Union

import cv2
import torch
import torchvision.transforms.functional as TF
from loguru import logger
from torch.utils.data import Dataset
from torchvision import transforms as T


class ColorDataset(Dataset):
    def __init__(self, root_dir: Path, crop_size: Optional[Union[Tuple, int]] = None, cut: float = 1.0):
        self.root_dir = root_dir
        self.files = list(root_dir.iterdir())
        self.files = self.files[:int(cut * len(self.files))]

        transforms = [
            T.Lambda(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
            T.ToTensor()
        ]

        if crop_size is not None:
            transforms.append(T.RandomCrop(crop_size, pad_if_needed=True))

        self.transform = T.Compose(transforms)

        logger.info(f'File[0]: {self.files[0]}, Total Files: {len(self.files)}')

    def __len__(self):
        return len(self.files)

    @logger.catch
    def __getitem__(self, index: int):
        try:
            rgb_img = self.transform(cv2.imread(str(self.files[index])))

            bw_img = TF.adjust_saturation(rgb_img, 0)

            return bw_img, rgb_img

        except Exception as e:
            logger.error(f'Exception at {self.files[index]}, {e}')
            return torch.tensor(-1), torch.tensor(-1)
