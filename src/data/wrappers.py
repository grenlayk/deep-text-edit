from typing import Tuple, List

import cv2
import numpy as np
from albumentations import ChannelShuffle, Normalize
from torch.utils.data import Dataset

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
        image = self.augment(image=image)["image"]
        data[self.image_key] = image

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

        self.norm = Normalize(mean=mean, std=std)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        for key in self.image_keys:
            image = data[key]
            image = self.norm(image=image)["image"]
            data[self.image_key] = image

        return data
