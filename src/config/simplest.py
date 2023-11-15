import torch
from pathlib import Path
import cv2
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader

from src.data.baseline import ImgurDataset
from src.data.wrappers import DrawText, ChannelShuffleImage, Resize, NormalizeImages, GetRandomText
from src.losses import VGGPerceptualLoss
from src.losses.ocr2 import OCRV2Loss
from src.models.rrdb import RRDB_pretrained, RRDBNet
from src.pipelines.simplest import SimplestEditing


class SimplestGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = RRDBNet(6, 3, 64, 10, gc=32)

    def forward(self, style, content):
        inputs = torch.concat([style, content], dim=1)
        res = self.backbone(inputs)
        return res


class Config:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    size = (192, 64)
    crops_path = Path("/cache/data/imgur/crops")

    batch_size = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self):
        generator = SimplestGenerator().to(self.device)
        optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)

        trainset = self.get_dataset(self.crops_path / 'train')
        valset = self.get_dataset(self.crops_path / 'val')

        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=12)
        self.valloader = DataLoader(valset, batch_size=self.batch_size)

        ocr = OCRV2Loss(self.mean, self.std).to(self.device)
        perc = VGGPerceptualLoss(self.mean, self.std).to(self.device)

        criterions = [
            {'criterion': ocr, 'name': 'ocr', 'pred_key': 'pred_base', 'target_key': 'draw_random'},
            {'criterion': perc, 'name': 'perc', 'pred_key': 'pred_base', 'target_key': 'image'},
        ]

        self.pipeline = SimplestEditing(
            generator=generator,
            optimizer=optimizer,
            criterions=criterions,
            style_key='image',
            draw_orig='draw_orig',
            text_orig='content',
            draw_rand='draw_random',
            text_rand='random',
        )

        self.trainer = Trainer(accelerator=self.device, max_epochs=10)

    def get_dataset(self, root):
        dataset = ImgurDataset(root)
        dataset = ChannelShuffleImage(dataset, 'image')

        dataset = GetRandomText(dataset, root, 'random')
        dataset = DrawText(dataset, 'content', 'draw_orig')
        dataset = DrawText(dataset, 'random', 'draw_random')

        dataset = Resize(dataset, 'image', self.size)
        dataset = Resize(dataset, 'draw_orig', self.size, interpolation=cv2.INTER_NEAREST)
        dataset = Resize(dataset, 'draw_random', self.size, interpolation=cv2.INTER_NEAREST)

        dataset = NormalizeImages(dataset, ['image', 'draw_orig', 'draw_random'], self.mean, self.std)
        return dataset

    def run(self):
        self.trainer.fit(self.pipeline, self.trainloader)
