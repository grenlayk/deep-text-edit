from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset


class BaselineDataset(Dataset):
    def __init__(self, root_dir: Path):
        '''
            root_dir - directory with 2 subdirectories - root_dir/style, root_dir/content
            Images in root_dir/content(hard-coded): 64x256
            Images in root_dir/style: arbitrary - need to be resized to 256x256
        '''
        self.root_dir = root_dir
        self.style_dir = root_dir / 'style'
        self.content_dir = root_dir / 'content' 
        self.style_files = list(self.style_dir.iterdir())
        self.content_files = list(self.content_dir.iterdir())

        logger.info(f'File[0]: {self.style_files[0]}, Total Files: {len(self.style_files) + len(self.content_files)}')

    def __len__(self):
        return len(self.style_files)

    def __getitem__(self, index):
        try:
            # *** Read the image from file ***
            img_style = cv2.imread(str(self.style_files[index]), cv2.IMREAD_COLOR)
            if img_style is None:
                raise Exception
            img_style = cv2.resize(img_style, (256, 256))
            img_style = img_style * 1.0 / 255
            img_style = torch.from_numpy(np.transpose(img_style[:, :, [2, 1, 0]], (2, 0, 1))).float()

            img_content = cv2.imread(str(self.content_files[index]), cv2.IMREAD_COLOR)
            if img_content is None:
                raise Exception
            img_style = cv2.resize(img_style, (256, 256))
            img_content = img_content * 1.0 / 255
            img_content = torch.from_numpy(np.transpose(img_content[:, :, [2, 1, 0]], (2, 0, 1))).float()

            return img_style, img_content, str(self.content_files[index].name.split('.', 1)[0])

        except Exception as e:
            logger.error(f'Exception at {self.style_files[index]}, {e}')
            return torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), 'Error'
