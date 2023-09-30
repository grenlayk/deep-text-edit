from typing import List, Dict, Any

import pytorch_lightning as pl
from torch import nn
from torch.optim import Optimizer


class SimplestEditing(pl.LightningModule):
    def __init__(
            self,
            generator: nn.Module,
            optimizer: Optimizer,
            criterions: List[Dict[str, Any]],
            style_key: str = 'image',
            original_key: str = 'draw',
            content_key: str = 'draw',
    ):
        super().__init__()
        self.generator = generator
        self.optimizer = optimizer
        self.criterions = criterions
        self.style_key = style_key
        self.original_key = original_key
        self.content_key = content_key

    def forward(self, style, content, postfix='base'):
        results = self.generator(style, content)
        results = {f"{key}_{postfix}": value for key, value in results.items()}
        return results

    def training_step(self, batch, batch_idx):
        style = batch[self.style_key]
        original = batch[self.original_key]
        content = batch[self.content_key]

        predictions = self.forward(style, content, 'base')
        predictions.update(self.forward(style, original, 'original'))
        predictions.update(batch)

        total = 0
        for criterion_dict in self.criterions:
            criterion, name, pred_key, target_key = [criterion_dict[key] for key in
                                                     ['criterion', 'name', 'pred_key', 'target_key']]
            loss = criterion(predictions[pred_key], predictions[target_key])
            self.log(name, loss)

            total = loss + total
        self.log('total', total)

        return total

    def configure_optimizers(self):
        return self.optimizer
