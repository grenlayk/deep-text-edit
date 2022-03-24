import cv2
import numpy as np
import torch
import json
import subprocess
import os
import numpy as np
import tarfile

from PIL import Image, ImageDraw, ImageFont
from loguru import logger
from torch.utils.data import Dataset
from src.disk import  disk
from pathlib import Path

def draw_one(text, dataset_folder: Path):
    img = Image.new('RGB', (256, 64), color = (255, 255, 255))
    fnt = ImageFont.truetype('/home/nikita/Huawei_Project/text-deep-fake/src/utils/VerilySerifMono.otf', 40)
    d = ImageDraw.Draw(img)
    d.text((60, 10), text, font=fnt, fill=(0, 0, 0))
    img.save(dataset_folder / '{}.png'.format('(' + text + ')'))


def download_data(remote_archieve_path: Path, local_dir: Path):
    '''
    Donwloads and unarchive  archive from disk(with remote_archieve_path path) to local_dir
    '''
    logger.info('Downloading data')
    local_path = local_dir / remote_archieve_path.name
    disk.download(str(remote_archieve_path), str(local_path))
    logger.info('Download finished, starting unarchivation')
    tarfile.open(local_path, 'r').extractall(local_dir)
    logger.info('Unarchieved')


def setup_dataset(style_dir: Path, content_dir: Path):
    '''
    Setup dataloader from BaselineDataset with style images from style_dir, and content images are drawn with text from json_path and saved to content_dir.
    Json's schema "image_path":"image_label"
    dataset_type must be either of ['train', 'test', 'val']
    '''

    json_path = style_dir / 'words.json'
    with open(json_path) as json_file:
        words = json.load(json_file)
    dataset_size = len(os.listdir(style_dir))
    if not content_dir.exists():
        logger.info("Drawing content pictures")
        idx = np.sort(np.random.choice(len(words), dataset_size, replace=False)) #Generate different words each time
        content_dir.mkdir()
        i = 0
        cur_idx = 0
        for _, text in words.items():
            if cur_idx >= dataset_size:
                break
            if i == idx[cur_idx]:
                if '/' in text:
                    continue
                draw_one(text, content_dir)
                cur_idx += 1
            i += 1
    
    return BaselineDataset(style_dir, content_dir)
        

class BaselineDataset(Dataset):
    def __init__(self, style_dir: Path, content_dir: Path):
        '''
            root_dir - directory with 2 subdirectories - root_dir/style, root_dir/content
            Images in root_dir/content(hard-coded): 64x256
            Images in root_dir/style: arbitrary - need to be resized to 256x256?
        '''
        self.style_dir = style_dir
        self.content_dir = content_dir 
        self.style_files = list(self.style_dir.iterdir())
        self.content_files = list(self.content_dir.iterdir())

        logger.info(f'File[0]: {self.style_files[0]}, Total Files: {len(self.style_files) + len(self.content_files)}')

    def __len__(self):
        return len(self.style_files)

    def __getitem__(self, index):
        try:
            img_style = cv2.imread(str(self.style_files[index]), cv2.IMREAD_COLOR)
            if img_style is None:
                raise Exception
            img_style = img_style * 1.0 / 255
            h, w, _ = img_style.shape
            img_style = torch.from_numpy(np.transpose(img_style[:, :, [2, 1, 0]], (2, 0, 1))).float()

            img_content = cv2.imread(str(self.content_files[index]), cv2.IMREAD_COLOR)
            if img_content is None:
                raise Exception
            img_content = cv2.resize(img_content, (w, h))
            img_content = img_content * 1.0 / 255
            img_content = torch.from_numpy(np.transpose(img_content[:, :, [2, 1, 0]], (2, 0, 1))).float()

            return img_style, img_content, str(self.content_files[index].name.split('.', 1)[0][1:-1])

        except Exception as e:
            logger.error(f'Exception at {self.style_files[index]}, {e}')
            return torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), 'Error'
