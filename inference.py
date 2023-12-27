from collections import OrderedDict

import click
import cv2
import numpy as np
import torch
from loguru import logger
from torch import nn

from src.models.style_brush import StypeBrush
from src.pipelines.utils import torch2numpy
from src.utils.draw import draw_word
from torchvision import transforms


@click.command()
@click.argument('checkpoint',
                type=click.Path(exists=True, dir_okay=False, readable=True),
                default='')
@click.argument('image_path',
                type=click.Path(exists=True, dir_okay=False, readable=True),
                default='')
@click.argument('text',
                default='')
@click.argument('save_path',
                type=click.Path(exists=True, dir_okay=False, readable=True),
                default='')
@logger.catch
def main(checkpoint, image_path, text, save_path):
    device = 'cuda'

    style = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)

    generator = StypeBrush()
    checkpoint = torch.load(checkpoint, map_location='cpu')
    state_dict = OrderedDict({key.replace('generator.', ''): value for key, value in checkpoint['state_dict'].items()})
    generator.load_state_dict(state_dict)
    generator = generator.to(device)
    generator = generator.eval()

    result = inference(generator, style, text, device)

    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), result)


def inference(generator: nn.Module, style_image: np.ndarray, text: str, device: str):
    assert style_image.dtype == np.uint8

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    size = (192, 64)

    content = draw_word(text)

    style = cv2.resize(style_image, size, interpolation=cv2.INTER_CUBIC)
    content = cv2.resize(content, size, interpolation=cv2.INTER_NEAREST)

    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(list(mean), list(std))])
    style = norm(style).to(device)
    content = norm(content).to(device)

    result = generator(style, content)
    result = torch2numpy(result, mean, std)

    return result


if __name__ == '__main__':
    main()
