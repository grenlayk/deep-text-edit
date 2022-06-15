import torch
from loguru import logger as info_logger
from src.disk import disk
from pathlib import Path
from src.logger.simple import Logger
from src.losses.vgg import VGGLoss
from src.data.baseline import BaselineDataset
from src.losses.vgg import VGGLoss
from src.utils.download import download_dataset
from src.models.stylegan import StyleBased_Generator
from src.training.stylegan import StyleGanTrainer
from src.storage.simple import Storage
from src.losses.STRFL import OCRLoss
from src.losses.typeface_perceptual import TypefacePerceptualLoss
from torchvision import models
from torchvision.models.resnet import BasicBlock
from torch.utils.data import DataLoader


class ContentResnet(models.ResNet):
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        return x


class Config:
    def __init__(self):
        disk.login()

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        info_logger.info(f'Using device: {device}')
        style_dir = Path('data/IMGUR5K')
        download_dataset('IMGUR5K')
        batch_size = 8
        train_dataloader = DataLoader(BaselineDataset(style_dir / 'train'), shuffle=True, batch_size=batch_size)
        val_dataloader = DataLoader(BaselineDataset(style_dir / 'val'), batch_size=batch_size)

        total_epochs = 500

        weights_folder = 'models/Stylegan (pretrained on content)'
        if not Path(weights_folder).exists():
            disk.download(weights_folder, weights_folder)

        model = StyleBased_Generator(dim_latent=512)
        model.load_state_dict(torch.load(f'{weights_folder}/model'))
        model.to(device)

        style_embedder = models.resnet18()
        style_embedder.fc = torch.nn.Identity()
        style_embedder = style_embedder.to(device)
        style_embedder.load_state_dict(torch.load(f'{weights_folder}/style_embedder'))

        content_embedder = ContentResnet(BasicBlock, [2, 2, 2, 2]).to(device)
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

        ocr_coef = 0.12
        cycle_coef = 0.25
        recon_coef = 0.25
        emb_coef = 4.0
        perc_coef = 100.0
        tex_coef = 5.0

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
