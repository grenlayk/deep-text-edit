from pathlib import Path
from typing import Union

import torch
from torch import nn


class Storage:
    def __init__(self, save_folder: Union[str, Path], save_freq: int = 1):
        if isinstance(save_folder, str):
            self.save_path = Path(save_folder)
        else:
            self.save_path = save_folder

        self.save_path.mkdir(parents=True)
        self.save_freq = save_freq
        self.epoch_count = 0

    def save(self, epoch: int, modules: dict[str, nn.Module], metric: float):
        '''
        modules dict example:

        {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler
        }
        '''

        self.epoch_count += 1
        if self.epoch_count == self.save_freq:
            self.epoch_count = 0
            # FIXME: We can only save nn.Module, but we need to save metrics too.
            # Should we transform the metric or change the typing?
            #
            # modules['metric'] = metric
            for module_name, module in modules.items():
                torch.save(module, self.save_path / module_name / str(epoch))
