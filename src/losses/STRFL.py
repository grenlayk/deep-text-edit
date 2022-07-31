# source: https://github.com/ku21fan/STR-Fewer-Labels
# 
# Copyright (c) 2021 Baek JeongHun
# 
# ------------------------------------------------
# loss function is modified to match our template

from loguru import logger
import torch
from torch import nn
from torchvision import transforms as T
import torch.nn.functional as F

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
            if not disk.get_disabled():
                disk.download(model_remote_path, model_local_path)
            else:
                logger.error('You need to download the TRBA-PR.pth from https://disk.yandex.ru/d/gTJa6Bg2QW0GJQ and '
                             'put it in the models/TRBA folder in the root of the repository')
                exit(1)

        opt = Options()
        self.opt = opt
        model = nn.DataParallel(TRBA(opt))
        model.load_state_dict(torch.load(model_local_path))
        self.model = model.module

        self.transform = resizeNormalize((img_h, img_w))

        self.converter = opt.Converter
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=opt.Converter.dict["[PAD]"])

        self.model.train()

    def forward(self, images, labels, return_recognized=False):
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

        if return_recognized:
            preds_size = torch.IntTensor([preds.size(1)] * batch_size).cuda()
            _, preds_index = preds.max(2)
            preds_str = self.opt.Converter.decode(preds_index, preds_size)
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            recognized = []

            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                pred_EOS = pred.find("[EOS]")
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]
                recognized.append(pred)

            return loss, recognized
        return loss
