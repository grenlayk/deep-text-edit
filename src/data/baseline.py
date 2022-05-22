import cv2
import numpy as np
import torch
import json
import random
import string

from PIL import Image, ImageDraw, ImageFont
from loguru import logger
from torch.utils.data import Dataset
from src.disk import  disk
from pathlib import Path

def draw_one(text: str):
<<<<<<< HEAD
    img = Image.new('RGB', (256, 64), color = (255, 255, 255))
    fnt = ImageFont.truetype('./data/VerilySerifMono.otf', 50)
    d = ImageDraw.Draw(img)
    text_width, text_height = d.textsize(text, fnt)
    position = ((256 - text_width) / 2, (64 - text_height) / 2)
=======
    text_len = len(text)
    font_size = 50
    w = max(128, int(text_len * font_size * 0.64))
    h = 64

    img = Image.new('RGB', (w, h), color = (255, 255, 255))
    fnt = ImageFont.truetype('./data/VerilySerifMono.otf', font_size)
    d = ImageDraw.Draw(img)
    text_width, text_height = d.textsize(text, fnt)
    position = ((w - text_width) / 2, (h - text_height) / 2)
>>>>>>> main

    d.text(position, text, font=fnt, fill = 0)
    return img        


class BaselineDataset(Dataset):
    def __init__(self, style_dir: Path):
        '''
            root_dir - directory with 2 subdirectories - root_dir/style, root_dir/content
            Images in root_dir/content(hard-coded): 64 x 256
            Images in root_dir/style: arbitrary - need to be resized to 256x256?
        '''
        self.style_dir = style_dir
        self.style_files = list(self.style_dir.iterdir())
        json_path = style_dir / 'words.json'
        with open(json_path) as json_file:
            self.words = json.load(json_file)
        logger.info(f'Total Files: {len(self.style_files) }')

    def __len__(self):
        return len(self.style_files)

    def __getitem__(self, index):
        try:
            img_size = (128, 64)
            if self.style_files[index] == self.style_dir / 'words.json':
                index = (index + 1) % len(self.style_files)
            img_style = cv2.imread(str(self.style_files[index]), cv2.IMREAD_COLOR)
            if img_style is None:
                raise Exception
<<<<<<< HEAD
            img_style = cv2.resize(img_style, (64, 64))
=======
            img_style = cv2.resize(img_style, img_size) 
>>>>>>> main
            img_style = img_style * 1.0 / 255
            img_style = torch.from_numpy(np.transpose(img_style[:, :, [2, 1, 0]], (2, 0, 1))).float()

            content = random.choice(list(self.words.values()))
            allowed_symbols = string.ascii_lowercase + string.digits
            content = ''.join([i for i in content if i in allowed_symbols])
            while not content:
                content = random.choice(list(self.words.values()))
                content = ''.join([i for i in content if i in allowed_symbols])
            pil_content = draw_one(content)
            img_content = np.array(pil_content)
<<<<<<< HEAD
            img_content = cv2.resize(img_content, (64, 64))

            img_content = img_content * 1.0 / 255
            img_content = torch.from_numpy(np.transpose(img_content[:, :, [2, 1, 0]], (2, 0, 1))).float()

=======
            img_content = cv2.resize(img_content, img_size) 

            img_content = img_content * 1.0 / 255
            img_content = torch.from_numpy(np.transpose(img_content[:, :, [2, 1, 0]], (2, 0, 1))).float()
>>>>>>> main

            return img_style, img_content, content

        except Exception as e:
            logger.error(f'Exception at {self.style_files[index]}, {e}')
            return torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), 'Error'
