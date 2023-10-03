import json
import os
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
from albumentations import ChannelShuffle
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.draw import draw_word


class DrawText(Dataset):
    def __init__(self, dataset: Dataset, text_key: str, draw_key: str):
        self.dataset = dataset
        self.text_key = text_key
        self.draw_key = draw_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        draw = draw_word(data[self.text_key])
        data[self.draw_key] = np.array(draw)
        return data


class DrawTextCache(Dataset):
    def __init__(self, dataset: Dataset, text_key: str, draw_key: str, folder: str):
        self.dataset = dataset
        self.text_key = text_key
        self.draw_key = draw_key
        self.folder = folder

        os.makedirs(folder, exist_ok=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]

        text = data[self.text_key]
        path = os.path.join(self.folder, f"{text}.png")
        if os.path.exists(path):
            draw = cv2.imread(path)
        else:
            draw = draw_word(data[self.text_key])
            cv2.imwrite(path, draw)

        data[self.draw_key] = np.array(draw)

        return data


class ChannelShuffleImage(Dataset):
    def __init__(self, dataset: Dataset, image_key: str):
        self.dataset = dataset
        self.image_key = image_key

        self.augment = ChannelShuffle()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]

        image = data[self.image_key]
        image = np.ascontiguousarray(self.augment(image=image)["image"]).copy()
        data[self.image_key] = image

        return data


class GetRandomText(Dataset):
    def __init__(self, dataset: Dataset, root: Path, key: str, max_text_len: int = 24):
        self.dataset = dataset
        self.root = root
        self.key = key
        self.max_text_len = max_text_len

        json_path = self.root / 'words.json'
        with open(json_path, 'r', encoding='utf-8') as json_file:
            words = json.load(json_file)

        allowed_symbols = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

        self.words = [word for word in words.values() if len(set(word) - set(allowed_symbols)) == 0 and word != '.']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]

        i = np.random.randint(len(self.words))

        text = self.words[i]
        if len(text) > self.max_text_len:
            text = text[:self.max_text_len]

        data[self.key] = text
        return data


class Resize(Dataset):
    def __init__(self, dataset: Dataset, image_key: str, size: Tuple[int, int], interpolation=cv2.INTER_CUBIC):
        self.dataset = dataset
        self.image_key = image_key
        self.size = size
        self.interpolation = interpolation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        image = data[self.image_key]
        resized = cv2.resize(image, self.size, interpolation=self.interpolation)
        data[self.image_key] = resized
        return data


class NormalizeImages(Dataset):
    def __init__(self, dataset: Dataset, image_keys: List[str], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.dataset = dataset
        self.image_keys = image_keys
        self.mean = mean
        self.std = std

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(list(self.mean), list(self.std))
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        for key in self.image_keys:
            image = data[key]
            image = self.norm(image)
            data[key] = image

        return data
