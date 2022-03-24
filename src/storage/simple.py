from pathlib import Path
from typing import Union

import torch
from src.disk import disk
from torch import nn


class Storage:
    def __init__(self, save_folder: Union[str, Path], save_freq: int = 1):
        if isinstance(save_folder, str):
            self.save_path = Path(save_folder)
        else:
            self.save_path = save_folder

        self.save_path.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq

    def save(self, epoch: int, modules: dict[str, nn.Module], metric: dict[str, float]):
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
            epoch_path.mkdir(parents=True, exist_ok=True)
            for module_name, module in modules.items():
                torch.save(module.state_dict(), epoch_path / module_name)
                disk.upload(epoch_path / module_name, epoch_path / module_name)
