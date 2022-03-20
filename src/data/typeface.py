import os
from pathlib import Path
import subprocess
import tarfile
from typing import Union
from loguru import logger
import numpy as np
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from src.disk import disk
from torchvision import transforms as T


class TypefaceDataset(ImageFolder):
    def __init__(self, remote: Union[str, Path], local: Union[str, Path]):
        if isinstance(remote, Path):
            remote_path = remote
        else:
            remote_path = Path(remote)
        if isinstance(local, Path):
            local_path = local
        else:
            local_path = Path(local)

        dir_path = Path(str(local_path).removesuffix(''.join(local_path.suffixes)))
        if not local_path.exists:
            logger.info('Downloading data')
            disk.download(remote_path, local_path)

            if len(local_path.suffixes) > 0:
                tool = local_path.suffix[1:]
                dir_path.mkdir()
                p = subprocess.run(
                    ['bash', '-c', f'tar -I {tool} -xvf {local_path} -C {dir_path}'],
                    capture_output=True
                )
                p.check_returncode()

        super().__init__(dir_path, T.Compose([
            T.ToTensor(),
            T.Resize((64, 196)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))

        logger.info('Dataset initialized')

    def _preprocess(self):
        raise NotImplementedError
