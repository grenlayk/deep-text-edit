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
    img = Image.new('L', (300, 64), color = 255)
    fnt = ImageFont.truetype('./data/utils/VerilySerifMono.otf', 40)
    d = ImageDraw.Draw(img)
    text_width, text_height = d.textsize(text, fnt)
    position = ((300-text_width)/2,(64-text_height)/2)

    d.text(position, text, font=fnt, fill = 0)
    return img        


class BaselineDataset(Dataset):
    def __init__(self, style_dir: Path):
        '''
            root_dir - directory with 2 subdirectories - root_dir/style, root_dir/content
            Images in root_dir/content(hard-coded): 64x256
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
            img_style = cv2.imread(str(self.style_files[index]), cv2.IMREAD_COLOR)
            if img_style is None:
                raise Exception
            img_style = cv2.resize(img_style, (128, 128))
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
            img_content = cv2.resize(img_content, (128, 128))
            img_content = img_content * 1.0 / 255
            img_content = torch.from_numpy(img_content).float().unsqueeze(0)

            return img_style, img_content, content

        except Exception as e:
            logger.error(f'Exception at {self.style_files[index]}, {e}')
            return torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), 'Error'
