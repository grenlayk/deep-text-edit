from pathlib import Path

import kornia
import torch
import torchvision
from kornia.enhance import Denormalize, Normalize
from loguru import logger

from src.disk import disk


class TypefacePerceptualLoss(torch.nn.Module):
    def __init__(
            self,
            mean,
            std,
            model_remote_path='models/TypefaceClassifier/model',
            model_local_path='models/TypefaceClassifier/model'):
        super().__init__()
        if not Path(model_local_path).exists():
            if not disk.get_disabled():
                disk.download(model_remote_path, model_local_path)
            else:
                logger.error(
                    'You need to download the TypefaceClassifier/model from https://disk.yandex.ru/d/gTJa6Bg2QW0GJQ and '
                    'put it in the models/ folder in the root of the repository')
                exit(1)

        model = torchvision.models.vgg16().cuda()
        model.classifier[-1] = torch.nn.Linear(4096, 2500)
        model.load_state_dict(torch.load(model_local_path))
        model.classifier[-1] = torch.nn.Identity()
        self.model = model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self.size = (224, 224)
        self.norm = Normalize(mean=[0.4803, 0.4481, 0.3976], std=[0.2769, 0.2690, 0.2820])
        self.denorm = Denormalize(torch.Tensor(mean), torch.Tensor(std))

    def prepare_sample(self, images):
        images = self.denorm(images)
        images = self.norm(images)
        images = kornia.geometry.transform.resize(images, self.size, interpolation=self.interpolation)
        return images

    def forward(self, inputs, targets):
        inputs = self.model(self.prepare_sample(inputs))
        targets = self.model(self.prepare_sample(targets))
        return torch.nn.functional.l1_loss(inputs, targets)
