from pathlib import Path
import cv2
from torch.utils.data import DataLoader

from src.data.baseline import ImgurDataset
from src.data.wrappers import DrawText, ChannelShuffleImage, Resize, NormalizeImages
from src.losses.ocr2 import STRFLInference

size = (192, 64)
root = Path("")
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

dataset = base = ImgurDataset(root)
dataset = ChannelShuffleImage(dataset, 'image')
dataset = DrawText(dataset, 'content', 'draw')
dataset = Resize(dataset, 'image', size)
dataset = Resize(dataset, 'draw', size, interpolation=cv2.INTER_NEAREST)
dataset = NormalizeImages(dataset, ['image', 'draw'], mean, std)

loader = DataLoader(dataset)

ocr = STRFLInference(mean, std)


for data in loader:
    break


