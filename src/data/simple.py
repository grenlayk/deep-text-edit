from pathlib import Path
from typing import Union
from torch.utils.data import Dataset
from src.disk import disk


class SimpleDataset(Dataset):
    def __init__(self, remote: Union[str, Path], local: Union[str, Path]):
        if isinstance(remote, Path):
            remote_path = remote
        else:
            remote_path = Path(remote)
        if isinstance(local, Path):
            local_path = local
        else:
            local_path = Path(local)

        if not local_path.exists():
            self._download(remote_path, local_path)

    def _preprocess(self):
        raise NotImplementedError

    def _download(self, remote_path: Path, local_path: Path):
        disk.download(remote_path, local_path)

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
