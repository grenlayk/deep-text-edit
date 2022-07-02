# source: https://github.com/meijieru/crnn.pytorch
# 
# Copyright (c) 2017 Jieru Mei meijieru@gmail.com

import collections

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from pathlib import Path
from src.models import ocr
from PIL import Image
from src.disk import disk

class resizeNormalize():
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, batch):
        batch = transforms.Resize(self.size)(batch)
        for n in range(batch.shape[0]):
            batch[n].sub_(0.5).div_(0.5)
        return batch

class strLabelConverter():
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l


class OCRLoss(nn.Module):
    def __init__(self, model_remote_path = 'ocr.pth', model_local_path = 'ocr.pth', alp = '0123456789abcdefghijklmnopqrstuvwxyz', hidden_state_size = 256, imH = 32, imW = 100):
        super().__init__()
        if not Path(model_local_path).exists():
            disk.download(model_remote_path, model_local_path)
        self.transform = resizeNormalize((imH, imW))
        self.ocr = ocr.crnn_pretrained(model_local_path, alp, hidden_state_size, imH)
        self.alp = alp
        self.converter = strLabelConverter(alp)
        self.criterion = torch.nn.CTCLoss(zero_infinity = True)

    def print_pred(self, res):
        _, preds = res.max(2)

        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = torch.IntTensor([preds.size(0)])
        raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        print('%-20s => %-20s' % (raw_pred, sim_pred))

    def forward(self, batch, labels):
        transformed_batch = self.transform(batch)
        gray_batch =  (0.299 * transformed_batch[:,0,:,:] + 0.587  * transformed_batch[:,1,:,:] + 0.114  * transformed_batch[:,2,:,:]).unsqueeze(1)
        texts, lengths = self.converter.encode(labels)
        res = self.ocr(gray_batch)
        #self.print_pred(res)
        sz = torch.tensor(res.size(0)).repeat(res.size(1))
        return self.criterion(res, texts, sz, lengths)       
