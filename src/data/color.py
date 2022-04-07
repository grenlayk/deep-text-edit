from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as TF
from loguru import logger
from torch.utils.data import Dataset
from torchvision import transforms as T


class CustomDataset(Dataset):
    def __init__(self, root_dir: Path, training: bool = False):
        self.root_dir = root_dir
        self.files = list(root_dir.iterdir())
        self.training = training
        logger.info(f'File[0]: {self.files[0]}, Total Files: {len(self.files)}')

    def __len__(self):
        return len(self.files)

    @logger.catch
    def __getitem__(self, index: int):
        try:
            rgb_img = T.ToTensor()(cv2.imread(str(self.files[index])))
            if rgb_img is None:
                raise Exception
            rgb_img = T.RandomCrop(32)(rgb_img)

            bw_img = TF.adjust_saturation(rgb_img, 0)

            return bw_img, rgb_img

        except Exception as e:
            logger.error(f'Exception at {self.files[index]}, {e}')
            return torch.tensor(-1), torch.tensor(-1)
