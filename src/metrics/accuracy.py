import torch


class Top1Accuracy:
    def __init__(self):
        pass

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, ) -> dict[str, torch.Tensor]:
        _, max_idx_class = pred.max(dim=1)
        return {'Acc@1': torch.mean((max_idx_class == target).float())}
