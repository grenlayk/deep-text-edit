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
            draw_orig: str = 'draw_orig',
            text_orig: str = 'content',
            draw_rand: str = 'draw_random',
            text_rand: str = 'random',
    ):
        super().__init__()
        self.generator = generator
        self.optimizer = optimizer
        self.criterions = criterions
        self.style_key = style_key
        self.draw_orig = draw_orig
        self.text_orig = text_orig
        self.draw_rand = draw_rand
        self.text_rand = text_rand

    def forward(self, style, content, postfix='base'):
        print('run')
        results = self.generator(style, content)
        if not isinstance(results, dict):
            results = {'pred': results}
        results = {f"{key}{postfix}": value for key, value in results.items()}
        return results

    def training_step(self, batch, batch_idx):
        style = batch[self.style_key]
        draw_orig = batch[self.draw_orig]
        draw_rand = batch[self.draw_rand]

        predictions = self.forward(style, draw_rand, '_base')
        predictions.update(self.forward(style, draw_orig, '_original'))
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
