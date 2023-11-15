from typing import List, Dict, Any, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.optim import Optimizer

from src.losses.ocr2 import STRFLInference
from src.pipelines.utils import torch2numpy
from src.utils.draw import draw_word


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

        self.ocr = STRFLInference(mean, std)

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

    def visualize_image(self, name, image):
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError('TensorBoard Logger not found')

        draw = image
        if isinstance(image, torch.Tensor):
            draw = torch2numpy(image, self.mean, self.std)

        if draw.shape[2] == 3:
            draw = draw.transpose((2, 0, 1)).copy()

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
            recogs_base = self.ocr.recognize(predictions['pred_original'])
            recogs_rand = self.ocr.recognize(predictions['pred_base'])
            for i in range(10):
                self.visualize_image(f'{i}/image', predictions[self.style_key][i])
                self.visualize_image(f'{i}/pred_base', predictions['pred_base'][i])
                self.visualize_image(f'{i}/pred_original', predictions['pred_original'][i])

                self.visualize_image(f'{i}/draw_orig', predictions[self.draw_orig][i])
                self.visualize_image(f'{i}/draw_rand', predictions[self.draw_rand][i])

                self.visualize_image(f'{i}/recog_orig', draw_word(recogs_base[i]))
                self.visualize_image(f'{i}/recog_rand', draw_word(recogs_rand[i]))

    def validation_epoch_end(self, outputs) -> None:
        for metric_dict in self.metrics:
            res = metric_dict['metric'].compute()
            self.log(metric_dict['name'], res)
            metric_dict['metric'].reset()

    def configure_optimizers(self):
        return self.optimizer


class SimplePerceptual(pl.LightningModule):
    def __init__(
            self,
            generator: nn.Module,
            generator_optimizer: Optimizer,
            criterions: List[Dict[str, Any]],
            metrics: List[Dict[str, Any]],
            style_key: str = 'image',
            draw_orig: str = 'draw_orig',
            text_orig: str = 'text_orig',
            draw_rand: str = 'draw_random',
            text_rand: str = 'text_random',
            gen_scheduler: Any = None,
            mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
            calc_orig: bool = True
    ):
        super().__init__()
        self.generator = generator
        self.generator_optimizer = generator_optimizer
        self.criterions = criterions
        self.metrics = metrics
        self.style_key = style_key
        self.draw_orig = draw_orig
        self.text_orig = text_orig
        self.draw_rand = draw_rand
        self.text_rand = text_rand
        self.gen_scheduler = gen_scheduler
        self.mean = mean
        self.std = std
        self.calc_orig = calc_orig

        self.ocr = STRFLInference(mean, std)
        self.automatic_optimization = False

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

        predictions = self.forward(style, draw_rand, '_random')
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

        self.generator_optimizer.zero_grad()
        self.manual_backward(total)
        self.generator_optimizer.step()
        self.generator_optimizer.zero_grad()

        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

        if batch_idx == 0:
            if self.calc_orig:
                recogs_base = self.ocr.recognize(predictions['pred_original'])
            recogs_rand = self.ocr.recognize(predictions['pred_random'])
            for i in range(10):
                self.visualize_image(f'train_{i}/image', predictions[self.style_key][i])
                self.visualize_image(f'train_{i}/pred_random', predictions['pred_random'][i])
                if self.calc_orig:
                    self.visualize_image(f'train_{i}/pred_original', predictions['pred_original'][i])

                self.visualize_image(f'train_{i}/draw_orig', predictions[self.draw_orig][i])
                self.visualize_image(f'train_{i}/draw_rand', predictions[self.draw_rand][i])

                if self.calc_orig:
                    self.visualize_image(f'train_{i}/recog_orig', draw_word(recogs_base[i]))
                self.visualize_image(f'train_{i}/recog_rand', draw_word(recogs_rand[i]))

    def visualize_image(self, name, image):
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError('TensorBoard Logger not found')

        draw = image
        if isinstance(image, torch.Tensor):
            draw = torch2numpy(image, self.mean, self.std)

        if draw.shape[2] == 3:
            draw = draw.transpose((2, 0, 1)).copy()

        tb_logger.add_image(name, draw, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        style = batch[self.style_key]
        draw_orig = batch[self.draw_orig]
        draw_rand = batch[self.draw_rand]

        predictions = self.forward(style, draw_rand, '_random')
        predictions.update(self.forward(style, draw_orig, '_original'))
        predictions.update(batch)

        # total = 0
        # for criterion_dict in self.criterions:
        #     criterion, name, pred_key, target_key = [criterion_dict[key] for key in
        #                                              ['criterion', 'name', 'pred_key', 'target_key']]
        #     loss = criterion(predictions[pred_key], predictions[target_key])
        #     self.log(f'val/{name}', loss)
        #
        #     total = loss + total
        # self.log('val/total', total)

        for metric_dict in self.metrics:
            metric, name, pred_key, target_key = [metric_dict[key] for key in
                                                  ['metric', 'name', 'pred_key', 'target_key']]
            metric.update(predictions[pred_key], predictions[target_key])

        if batch_idx == 0:
            if self.calc_orig:
                recogs_base = self.ocr.recognize(predictions['pred_original'])
            recogs_rand = self.ocr.recognize(predictions['pred_random'])
            for i in range(10):
                self.visualize_image(f'val_{i}/image', predictions[self.style_key][i])
                self.visualize_image(f'val_{i}/pred_random', predictions['pred_random'][i])
                if self.calc_orig:
                    self.visualize_image(f'val_{i}/pred_original', predictions['pred_original'][i])

                self.visualize_image(f'val_{i}/draw_orig', predictions[self.draw_orig][i])
                self.visualize_image(f'val_{i}/draw_rand', predictions[self.draw_rand][i])

                if self.calc_orig:
                    self.visualize_image(f'val_{i}/recog_orig', draw_word(recogs_base[i]))
                self.visualize_image(f'val_{i}/recog_rand', draw_word(recogs_rand[i]))

    def validation_epoch_end(self, outputs) -> None:
        for metric_dict in self.metrics:
            res = metric_dict['metric'].compute()
            self.log(metric_dict['name'], res)
            metric_dict['metric'].reset()

    def configure_optimizers(self):

        # schedulers = []
        # if self.gen_scheduler is not None:
        #     gen_scheduler = {'scheduler': self.gen_scheduler, 'interval': 'step', 'frequency': 1}
        #     disc_scheduler = {'scheduler': self.disc_scheduler, 'interval': 'step', 'frequency': 1}
        #     schedulers = [gen_scheduler, disc_scheduler]

        return self.generator_optimizer
