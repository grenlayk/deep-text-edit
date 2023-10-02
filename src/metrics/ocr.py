import torch
from torchmetrics.text import CharErrorRate

from src.losses.ocr2 import STRFLInference


class ImageCharErrorRate(CharErrorRate):
    def __init__(self, mean, std, model_local_path='models/TRBA/TRBA-PR.pth'):
        super().__init__()

        self.ocr = STRFLInference(mean, std, model_local_path)

    def update(self, preds: torch.Tensor, target: Union[str, List[str]]) -> None:
        results = self.ocr.recognize(preds)

        super().update(results, target)
