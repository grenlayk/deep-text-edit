import torch
from pathlib import Path
import cv2
from torch.utils.data import DataLoader

from src.data.baseline import ImgurDataset
from src.data.wrappers import DrawText, ChannelShuffleImage, Resize, NormalizeImages
from src.models.rrdb import RRDB_pretrained, RRDBNet
from src.pipelines.simplest import SimplestEditing


class Config:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    size = (192, 64)
    crops_path = Path("")

    batch_size = 64

    def __init__(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        generator = RRDBNet(6, 3, 64, 10, gc=32).to(device)
        optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)

        trainset = self.get_dataset(self.crops_path / 'train')
        valset = self.get_dataset(self.crops_path / 'val')

        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        self.valloader = DataLoader(valset, batch_size=self.batch_size, shuffle=True)

        self.pipeline = SimplestEditing(
            generator=generator,
            optimizer=optimizer,

        )

    def get_dataset(self, root):
        dataset = ImgurDataset(root)
        dataset = ChannelShuffleImage(dataset, 'image')
        dataset = DrawText(dataset, 'content', 'draw')
        dataset = Resize(dataset, 'image', self.size)
        dataset = Resize(dataset, 'draw', self.size, interpolation=cv2.INTER_NEAREST)
        dataset = NormalizeImages(dataset, ['image', 'draw'], self.mean, self.std)
        return dataset
