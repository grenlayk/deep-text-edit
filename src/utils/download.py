from loguru import logger 
from pathlib import Path
from src.disk import disk

import tarfile

def download_data(remote_archieve_path: Path, local_dir: Path):
    '''
    Donwloads and unarchive  archive from disk(with remote_archieve_path path) to local_dir
    '''
    logger.info('Downloading data')
    local_path = local_dir / remote_archieve_path.name
    disk.download(str(remote_archieve_path), str(local_path))
    logger.info('Download finished')
    return local_path

def unarchieve(local_path: Path):
    logger.info('Unarchieved')
    tarfile.open(local_path, 'r').extractall(local_path.parent)
    logger.info('Unarchieved')