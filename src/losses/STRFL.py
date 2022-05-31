import torch
from torch import nn
from torchvision import transforms as T

from pathlib import Path
from src.models.STRFL import TRBA, Options
from PIL import Image
from src.disk import disk


class resizeNormalize():
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = T.ToTensor()

    def __call__(self, batch):
        batch = T.Resize(self.size)(batch)
        for n in range(batch.shape[0]):
            batch[n].sub_(0.5).div_(0.5)
        return batch


class OCRLoss(nn.Module):
    def __init__(self,
                 model_remote_path: str = 'models/TRBA/TRBA-PR.pth',
                 model_local_path: str = 'models/TRBA/TRBA-PR.pth'):
        super().__init__()

        img_h = 32
        img_w = 100

        if not Path(model_local_path).exists():
            disk.download(model_remote_path, model_local_path)

        opt = Options()
        self.opt = opt
        model = nn.DataParallel(TRBA(opt))
        model.load_state_dict(torch.load(model_local_path))
        self.model = model.module

        self.transform = resizeNormalize((img_h, img_w))

        self.converter = opt.Converter
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=opt.Converter.dict["[PAD]"])

        self.model.train()

    def forward(self, images, labels):
        batch_size = images.size(0)
        # For max length prediction
        text_for_pred = (
            torch.LongTensor(batch_size)
            .fill_(self.opt.Converter.dict["[SOS]"])
            .cuda()
        )
        labels_index, labels_length = self.opt.Converter.encode(
            labels, batch_max_length=25
        )
        target = labels_index[:, 1:]  # without [SOS] Symbol

        preds = self.model(self.transform(images), text_for_pred, is_train=False)
        loss = self.criterion(
            preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
        )

        return loss
