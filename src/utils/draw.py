import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T


def img_to_tensor(img: Image) -> torch.Tensor:
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((64, 192))
    ])
    return transform(img)


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
