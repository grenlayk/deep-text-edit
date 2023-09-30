# source: https://github.com/ku21fan/STR-Fewer-Labels
# 
# Copyright (c) 2021 Baek JeongHun
# 
# ------------------------------------------------
# loss function is modified to match our template

import kornia
import torch
import torch.nn.functional as F
from kornia.enhance import Denormalize
from torch import nn

from src.models.STRFL import TRBA, Options


class resizeNormalize():
    def __init__(self, size, interpolation='bicubic'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, batch):
        batch = kornia.geometry.transform.resize(batch, self.size, interpolation=self.interpolation)
        batch = (batch - 0.5) / 0.5
        return batch


class STRFLInference(nn.Module):
    def __init__(
            self,
            mean,
            std,
            model_local_path: str = 'models/TRBA/TRBA-PR.pth',
            img_h=32,
            img_w=100
    ):
        super().__init__()

        self.denorm = Denormalize(torch.Tensor(mean), torch.Tensor(std))

        opt = Options()
        self.opt = opt
        model = nn.DataParallel(TRBA(opt))
        model.load_state_dict(torch.load(model_local_path))
        self.model = model.module

        self.transform = resizeNormalize((img_h, img_w))

        self.converter = opt.Converter
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=opt.Converter.dict["[PAD]"])

        self.model.eval()

    def forward(self, images, is_train=False):
        images1 = self.denorm(images)
        images2 = self.transform(images1)

        batch_size = images2.size(0)
        text_for_pred = torch.LongTensor(batch_size).fill_(self.opt.Converter.dict["[SOS]"]).to(images2.device)

        preds = self.model(images2, text_for_pred, is_train=is_train)

        return preds

    def postprocess(self, preds):
        batch_size = preds.shape[0]
        preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(preds.device)
        _, preds_index = preds.max(2)
        preds_str = self.opt.Converter.decode(preds_index, preds_size)

        recognized = []
        for pred in preds_str:
            pred_EOS = pred.find("[EOS]")
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            recognized.append(pred)

        return recognized

    def recognize(self, images):
        preds = self.forward(images, is_train=False)

        recognized = self.postprocess(preds)

        return recognized


class OCRV2Loss(nn.Module):
    def __init__(
            self,
            mean,
            std,
            model_local_path: str = 'models/TRBA/TRBA-PR.pth',
            img_h=32,
            img_w=100
    ):
        super().__init__()

        self.model = STRFLInference(mean, std, model_local_path, img_h, img_w)

        opt = Options()
        self.opt = opt
        self.converter = opt.Converter
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=opt.Converter.dict["[PAD]"])

        self.model.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        import pdb
        pdb.set_trace()

    def forward(self, images, labels, return_recognized=False):
        preds = self.model.forward(images, is_train=False)

        labels_index, labels_length = self.opt.Converter.encode(labels, batch_max_length=25)
        target = labels_index[:, 1:]  # without [SOS] Symbol

        loss = self.criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        if return_recognized:
            recognized = self.postprocess(preds)
            return loss, recognized

        return loss
