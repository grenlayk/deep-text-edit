from loguru import logger
import torch
import torchvision
import torchvision.transforms as T
from src.disk import disk
from pathlib import Path


class TypefacePerceptualLoss(torch.nn.Module):
    def __init__(
            self,
            model_remote_path='models/TypefaceClassifier/model',
            model_local_path='models/TypefaceClassifier/model'):
        super().__init__()
        if not Path(model_local_path).exists():
            if not disk.get_disabled():
                disk.download(model_remote_path, model_local_path)
            else:
                logger.error('You need to download the TypefaceClassifier/model from https://disk.yandex.ru/d/gTJa6Bg2QW0GJQ and '
                             'put it in the models/ folder in the root of the repository')
                exit(1)

        model = torchvision.models.vgg16().cuda()
        model.classifier[-1] = torch.nn.Linear(4096, 2500)
        model.load_state_dict(torch.load(model_local_path))
        model.classifier[-1] = torch.nn.Identity()
        self.model = model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self.transform = T.Compose([
            T.Resize((64, 192)),
            T.Normalize([0.4803, 0.4481, 0.3976], [0.2769, 0.2690, 0.2820]),
        ])

    def forward(self, inputs, targets):
        return torch.nn.functional.l1_loss(self.model(inputs), self.model(targets))
