import shutil
from pathlib import Path
from typing import Union, Dict

import click
import torch
from src.disk import disk
from torch import nn


class Storage:
    def __init__(self, save_folder: Union[str, Path], save_freq: int = 1):
        self.save_path = Path(save_folder)

        if self.save_path.exists():
            click.confirm(f'Path {self.save_path} already exists. Discard its contents?', abort=True)
            if self.save_path.is_dir():
                shutil.rmtree(self.save_path)
            else:
                self.save_path.unlink()

        self.save_path.mkdir(parents=True)
        self.save_freq = save_freq

    def save(self, epoch: int, modules: Dict[str, nn.Module], metric: Dict[str, float]):
        '''
        modules dict example:

        {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler
        }
        '''
        if epoch % self.save_freq == 0:
            epoch_path = self.save_path / str(epoch)
            epoch_path.mkdir()
            for module_name, module in modules.items():
                torch.save(module.state_dict(), epoch_path / module_name)
                disk.upload(epoch_path / module_name, epoch_path / module_name)
