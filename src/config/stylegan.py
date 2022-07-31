import torch
from loguru import logger as info_logger
from src.disk import disk
from pathlib import Path
from src.logger.simple import Logger
from src.losses.vgg import VGGLoss
from src.data.baseline import BaselineDataset
from src.losses.vgg import VGGLoss
from src.utils.download import download_dataset
from src.models.embedders import ContentResnet, StyleResnet
from src.models.stylegan import StyleBased_Generator
from src.training.stylegan import StyleGanTrainer
from src.storage.simple import Storage
from src.losses.STRFL import OCRLoss
from src.losses.typeface_perceptual import TypefacePerceptualLoss
from torch.utils.data import DataLoader


class Config:
    def __init__(self):
        disk.login()

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        info_logger.info(f'Using device: {device}')
        style_dir = Path('data/IMGUR5K')
        if not disk.get_disabled():
            download_dataset('IMGUR5K')
        else:
            if not Path(style_dir).exists():
                logger.error('You should download IMGUR5K dataset first.')
                exit(1)
        batch_size = 16
        train_dataloader = DataLoader(BaselineDataset(style_dir / 'train'), shuffle=True, batch_size=batch_size)
        val_dataloader = DataLoader(BaselineDataset(style_dir / 'val'), batch_size=batch_size)

        total_epochs = 500

        weights_folder_name = 'Stylegan (pretrained on content)'
        weights_folder = f'models/{weights_folder_name}'
        if not Path(weights_folder).exists():
            if not disk.get_disabled():
                disk.download(weights_folder, weights_folder)
            else:
                logger.error(f'You need to download the {weights_folder_name} folder from '
                             'https://disk.yandex.ru/d/gTJa6Bg2QW0GJQ and '
                             'put it in the models/ folder in the root of the repository')
                exit(1)

        model = StyleBased_Generator(dim_latent=512)
        model.load_state_dict(torch.load(f'{weights_folder}/model'))
        model.to(device)

        style_embedder = StyleResnet().to(device) 
        style_embedder.load_state_dict(torch.load(f'{weights_folder}/style_embedder'))

        content_embedder = ContentResnet().to(device)
        content_embedder.load_state_dict(torch.load(f'{weights_folder}/content_embedder'))

        optimizer = torch.optim.AdamW(
            list(model.parameters()) +
            list(style_embedder.parameters()) +
            list(content_embedder.parameters()),
            lr=1e-3,
            weight_decay=1e-6
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.8
        )

        ocr_coef = 0.07
        cycle_coef = 2.0
        recon_coef = 2.0
        emb_coef = 0.0
        perc_coef = 0.0
        tex_coef = 6.0

        storage = Storage('checkpoints/stylegan(pretrained_on_content)_typeface_ocr_192x64')

        logger = Logger(
            image_freq=100,
            project_name='TDF',
            config={
                'ocr_coef': ocr_coef,
                'cycle_coef': cycle_coef,
                'recon_coef': recon_coef,
                'emb_coef': emb_coef,
                'perc_coef': perc_coef,
                'tex_coef': tex_coef,
                'img_size': (192, 64)
            }
        )

        self.trainer = StyleGanTrainer(
            model,
            style_embedder,
            content_embedder,
            optimizer,
            scheduler,
            train_dataloader,
            val_dataloader,
            storage,
            logger,
            total_epochs,
            device,
            ocr_coef,
            cycle_coef,
            recon_coef,
            emb_coef,
            perc_coef,
            tex_coef,
            OCRLoss(),
            TypefacePerceptualLoss(),
            VGGLoss(),
            torch.nn.L1Loss()
        )

    def run(self):
        self.trainer.run()
