from typing import List, Dict, Any, Tuple

import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.optim import Optimizer

from src.pipelines.utils import torch2numpy, add_text_to_image


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
        # predictions.update(self.forward(style, draw_orig, '_original'))
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


class SimplestEditingVal(pl.LightningModule):
    def __init__(
            self,
            generator: nn.Module,
            optimizer: Optimizer,
            criterions: List[Dict[str, Any]],
            metrics: List[Dict[str, Any]],
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
        self.metrics = metrics
        self.style_key = style_key
        self.draw_orig = draw_orig
        self.text_orig = text_orig
        self.draw_rand = draw_rand
        self.text_rand = text_rand

    def forward(self, style, content, postfix='base'):
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
        # predictions.update(self.forward(style, draw_orig, '_original'))
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

    def validation_step(self, batch, batch_idx):
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
            self.log(f'val/{name}', loss)

            total = loss + total
        self.log('val/total', total)

        for metric_dict in self.metrics:
            metric, name, pred_key, target_key = [metric_dict[key] for key in
                                                  ['metric', 'name', 'pred_key', 'target_key']]
            metric.update(predictions[pred_key], predictions[target_key])

    def validation_epoch_end(self, outputs) -> None:
        for metric_dict in self.metrics:
            res = metric_dict['metric'].compute()
            self.log(metric_dict['name'], res)
            metric_dict['metric'].reset()

    def configure_optimizers(self):
        return self.optimizer


class SimplestEditingViz(pl.LightningModule):
    def __init__(
            self,
            generator: nn.Module,
            optimizer: Optimizer,
            criterions: List[Dict[str, Any]],
            metrics: List[Dict[str, Any]],
            style_key: str = 'image',
            draw_orig: str = 'draw_orig',
            text_orig: str = 'content',
            draw_rand: str = 'draw_random',
            text_rand: str = 'random',
            mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.generator = generator
        self.optimizer = optimizer
        self.criterions = criterions
        self.metrics = metrics
        self.style_key = style_key
        self.draw_orig = draw_orig
        self.text_orig = text_orig
        self.draw_rand = draw_rand
        self.text_rand = text_rand
        self.mean = mean
        self.std = std

    def forward(self, style, content, postfix='base'):
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
        # predictions.update(self.forward(style, draw_orig, '_original'))
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

    def visualize_image(self, name, image, text=None):
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError('TensorBoard Logger not found')

        draw = torch2numpy(image, self.mean, self.std)
        if text is not None:
            draw = add_text_to_image(draw, text, font_color=(0, 255, 0))

        tb_logger.add_image(name, draw, self.current_epoch)

    def validation_step(self, batch, batch_idx):
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
            self.log(f'val/{name}', loss)

            total = loss + total
        self.log('val/total', total)

        for metric_dict in self.metrics:
            metric, name, pred_key, target_key = [metric_dict[key] for key in
                                                  ['metric', 'name', 'pred_key', 'target_key']]
            metric.update(predictions[pred_key], predictions[target_key])


        if batch_idx == 0:
            for i in range(10):
                self.visualize_image()

    def validation_epoch_end(self, outputs) -> None:
        for metric_dict in self.metrics:
            res = metric_dict['metric'].compute()
            self.log(metric_dict['name'], res)
            metric_dict['metric'].reset()

    def configure_optimizers(self):
        return self.optimizer
