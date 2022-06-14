import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def img_to_tensor(img: Image) -> torch.Tensor:
    return torch.from_numpy(
        np.transpose(
                    (cv2.resize(np.array(img), (192, 64)) / 255)[:, :, [2, 1, 0]], (2, 0, 1)
        )
    ).float()


def draw_word(word: str) -> Image:
    text_len = len(word)
    font_size = 50
    w = max(192, int(text_len * font_size * 0.64))
    h = 64

    img = Image.new('RGB', (w, h), color=(255, 255, 255))
    fnt = ImageFont.truetype('./data/VerilySerifMono.otf', font_size)
    d = ImageDraw.Draw(img)
    text_width, text_height = d.textsize(word, fnt)
    position = ((w - text_width) / 2, (h - text_height) / 2)

    d.text(position, word, font=fnt, fill=0)
    return img
