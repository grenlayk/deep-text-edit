from pathlib import Path
from src.disk import disk

import os
import tarfile


def is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)

    prefix = os.path.commonprefix([abs_directory, abs_target])

    return prefix == abs_directory


def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")

    tar.extractall(path, members, numeric_owner=numeric_owner)


def download_dataset(name: str):
    '''
    Donwloads and unarchive  archive from disk(with remote_archieve_path path) to local_dir
    '''
    if Path(f'data/{name}').exists():
        return

    disk.download(f'data/{name}.tar', f'data/{name}.tar')
    with tarfile.open(f'data/{name}.tar', 'r') as tar:
        safe_extract(tar, f'data/{name}')
