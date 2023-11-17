from pathlib import Path

import cv2
import torch
import torchmetrics.image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import L1Loss
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.baseline import ImgurDataset
from src.data.wrappers import ChannelShuffleImage, Resize, NormalizeImages, GetRandomText, DrawTextCache
from src.losses import VGGPerceptualLoss
from src.losses.lsgan import LSGeneratorCriterion, LSDiscriminatorCriterion
from src.losses.ocr2 import OCRV2Loss
from src.losses.utils import LossScaler
from src.metrics.ocr import ImageCharErrorRate
from src.models.nlayer_discriminator import NLayerDiscriminator
from src.models.style_brush import StypeBrush
from src.pipelines.brush import TextBrushPipeline
from src.utils.warmup import WarmupScheduler


class Config:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    size = (192, 64)
    crops_path = Path("/cache/data/imgur/crops")

    batch_size = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self):
        generator = StypeBrush().to(self.device)
        discriminator = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3,
                                            norm_layer=(lambda x: torch.nn.Identity())).to(self.device)
        generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=2e-3, betas=(0.5, 0.99))
        discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.99), eps=1e-8)

        trainset = self.get_dataset(self.crops_path / 'train')
        valset = self.get_dataset(self.crops_path / 'val')

        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        self.valloader = DataLoader(valset, batch_size=self.batch_size)

        perc = VGGPerceptualLoss(self.mean, self.std, feature_layers=(1, 2, 3), style_layers=()).to(self.device)
        style = VGGPerceptualLoss(self.mean, self.std, feature_layers=(), style_layers=(1, 2, 3)).to(self.device)
        style = LossScaler(style, 0.0005)
        l1 = L1Loss()
        ocr = OCRV2Loss(self.mean, self.std).to(self.device)

        criterions = [
            {'criterion': perc, 'name': 'train/rec_perc', 'pred_key': 'pred_original', 'target_key': 'image'},
            {'criterion': l1, 'name': 'train/rec_l1', 'pred_key': 'pred_original', 'target_key': 'image'},

            {'criterion': perc, 'name': 'train/cyc_perc', 'pred_key': 'pred_cycle', 'target_key': 'image'},
            {'criterion': l1, 'name': 'train/cyc_l1', 'pred_key': 'pred_cycle', 'target_key': 'image'},

            {'criterion': style, 'name': 'train/gram', 'pred_key': 'pred_random', 'target_key': 'image'},

            {'criterion': ocr, 'name': 'train/orig_ocr', 'pred_key': 'pred_cycle', 'target_key': 'text_original'},
            {'criterion': ocr, 'name': 'train/rand_ocr', 'pred_key': 'pred_random', 'target_key': 'text_random'},
        ]

        gen_loss = LSGeneratorCriterion()
        disc_loss = LSDiscriminatorCriterion()

        g_criterions = [
            {'criterion': gen_loss, 'name': 'train/gen_rand', 'real': 'image', 'fake': 'pred_random'},
            {'criterion': gen_loss, 'name': 'train/gen_orig', 'real': 'image', 'fake': 'pred_original'},
            {'criterion': gen_loss, 'name': 'train/gen_cyc', 'real': 'image', 'fake': 'pred_cycle'},
        ]
        d_criterions = [
            {'criterion': disc_loss, 'name': 'train/disc_rand', 'real': 'image', 'fake': 'pred_random'},
            {'criterion': disc_loss, 'name': 'train/disc_orig', 'real': 'image',
             'fake': 'pred_original'},
            {'criterion': disc_loss, 'name': 'train/disc_cyc', 'real': 'image', 'fake': 'pred_cycle'},

        ]

        cer = ImageCharErrorRate(self.mean, self.std).to(self.device)
        psnr = torchmetrics.image.PeakSignalNoiseRatio().to(self.device)

        metrics = [
            {'metric': cer, 'name': 'val/cer', 'pred_key': 'pred_random', 'target_key': 'text_random'},
            {'metric': psnr, 'name': 'val/psnr', 'pred_key': 'pred_cycle', 'target_key': 'image'},
        ]

        warmup = 5000
        gen_sch = SequentialLR(
            generator_optimizer,
            [WarmupScheduler(generator_optimizer, warmup), CosineAnnealingLR(generator_optimizer, 1000000)],
            [warmup]
        )

        self.pipeline = TextBrushPipeline(
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            criterions=criterions,
            g_criterions=g_criterions,
            d_criterions=d_criterions,
            metrics=metrics,
            style_key='image',
            draw_orig='draw_original',
            text_orig='text_original',
            draw_rand='draw_random',
            text_rand='text_random',
            gen_scheduler=gen_sch,
            mean=self.mean,
            std=self.std,
        )

        tb_path = Path("lightning_logs/tensorboard") / Path(__file__).stem
        checkpoint_path = Path("lightning_logs/checkpoint") / Path(__file__).stem
        logger = TensorBoardLogger(str(tb_path))
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, save_last=True)

        self.trainer = Trainer(logger=logger, callbacks=[LearningRateMonitor(), checkpoint_callback],
                               accelerator=self.device, max_epochs=200)

    def get_dataset(self, root):
        dataset = ImgurDataset(root, text_key='text_original')
        dataset = ChannelShuffleImage(dataset, 'image')

        dataset = GetRandomText(dataset, root, 'text_random')
        draw_cache = "/cache/data/cache/draw"
        dataset = DrawTextCache(dataset, 'text_original', 'draw_original', draw_cache)
        dataset = DrawTextCache(dataset, 'text_random', 'draw_random', draw_cache)

        dataset = Resize(dataset, 'image', self.size)
        dataset = Resize(dataset, 'draw_original', self.size, interpolation=cv2.INTER_NEAREST)
        dataset = Resize(dataset, 'draw_random', self.size, interpolation=cv2.INTER_NEAREST)

        dataset = NormalizeImages(dataset, ['image', 'draw_original', 'draw_random'], self.mean, self.std)
        return dataset

    def run(self):
        self.trainer.fit(self.pipeline, self.trainloader, self.valloader)
