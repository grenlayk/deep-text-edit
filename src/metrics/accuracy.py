import torch
from typing import Dict, Tuple, Union


# source:
# https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b?permalink_comment_id=3662283#gistcomment-3662283
class TopKAccuracy:
    def __init__(self, k: Union[int, Tuple]):
        self.k: Tuple
        if isinstance(k, int):
            self.k = (k,)
        else:
            self.k = tuple(k)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            maxk = max(self.k)
            batch_size = target.size(0)

            # get top maxk indicies that correspond to the most likely probability scores
            # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
            _, y_pred = pred.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
            # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
            y_pred = y_pred.t()

            # - get the credit for each example if the models predictions is in maxk values (main crux of code)
            # for any example, the model will get credit if it's prediction matches the ground truth
            # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
            # if the k'th top answer of the model matches the truth we get 1.
            # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
            target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
            # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
            # [maxk, B] were for each example we know which topk prediction matched truth
            correct = (y_pred == target_reshaped)
            # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

            # get topk accuracy
            topk_accs = {}  # idx is topk1, topk2, ... etc
            for k in self.k:
                # get tensor of which topk answer was right
                ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
                # flatten it to help compute if we got it correct for each example in batch
                # [k, B] -> [kB]
                flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()
                # get if we got it right for any of our top k prediction for each example in batch
                tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0,
                                                                                            keepdim=True)  # [kB] -> [1]
                # compute topk accuracy - the accuracy of the mode's ability to get it
                # right within it's top k guesses/preds
                topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
                topk_accs[f'Top-{k} Acc'] = topk_acc.item() * 100
            return topk_accs
