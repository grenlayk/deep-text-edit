from pathlib import Path
import cv2
from src.data.baseline import ImgurDataset
from src.data.wrappers import DrawText, ChannelShuffleImage, Resize, NormalizeImages

size = (192, 64)
root = Path("")

dataset = base = ImgurDataset(root)
dataset = ChannelShuffleImage(dataset, 'image')
dataset = DrawText(dataset, 'content', 'draw')
dataset = Resize(dataset, 'image', size)
dataset = Resize(dataset, 'draw', size, interpolation=cv2.INTER_NEAREST)
dataset = NormalizeImages(dataset, ['image', 'draw'])

