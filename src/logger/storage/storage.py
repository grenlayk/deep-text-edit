import datetime
from pathlib import Path
from __future__ import annotations
import torch

class Storage:
    def __init__(self, save_folder: str, save_freq: int):
        self.save_folder = Path(save_folder) / datetime.datetime.utcnow().strftime('%Y-%m-%d %H-%M-%S')
        self.save_folder.mkdir()
        self.save_freq = save_freq
        self.epoch_count = 0
    
    def save(self, epoch: int, modules: dict[str, dict], metric: float):
        ''' 
            modules dict example: 
            {'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()}
        '''
        self.epoch_count += 1
        if self.epoch_count == self.save_freq:
            self.epoch_count = 0
            modules['metric'] = metric
            torch.save(modules, self.save_folder / str(epoch))