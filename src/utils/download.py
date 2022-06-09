from pathlib import Path
from src.disk import disk

import tarfile


def download_dataset(name: str):
    '''
    Donwloads and unarchive  archive from disk(with remote_archieve_path path) to local_dir
    '''
    if Path(f'data/{name}').exists():
        return

    disk.download(f'data/{name}.tar', f'data/{name}.tar')
    with tarfile.open(f'data/{name}.tar', 'r') as tar:
        tar.extractall(f'data/{name}')
