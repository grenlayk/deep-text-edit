from typing import List

import torch


class ComposeLoss(torch.nn.Module):
    def __init__(self, losses: List, coefs: List[float]):
        super().__init__()
        assert len(losses) == len(coefs)
        self._losses = losses
        self._coefs = coefs

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        results = [f(pred, target) for f in self._losses]
        return {
            'total': sum(result * coef for result, coef in zip(results, self._coefs)),
            **{
                type(loss).__name__: result * coef for loss, coef, result in zip(self._losses, self._coefs, results)
            }
        }
