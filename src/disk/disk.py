import configparser
from pathlib import Path
from typing import List, Optional, Union

import yadisk
from loguru import logger

class Disk:
    '''
    Disk is a class for working with the folder in YandexDisk. To use an instance
    of this class, you must first perform the login (`Disk.login`).
    '''

    def __init__(self):
        self._logged_in = False

        # The credentials.ini file contains the app secret and token from the
        # previous login to not prompt the user each time.
        self._cred_cache_path = Path('.yadisk_cache/credentials.ini')

        self._cred_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._disabled = True

    def set_disabled(self, disable):
        self._disabled = disable

    def get_disabled(self):
        return self._disabled

    def login(self, use_cache=True, cache_credentials=True):
        '''
        Login to Yandex Disk. You must call this method before you can use other functions.
        '''
        if self._disabled:
            return

        if use_cache and self._cred_cache_path.exists():
            config = configparser.ConfigParser()
            config.read(self._cred_cache_path, encoding='utf-8')
            secret = config['YaDiskCreds']['secret']
            token = config['YaDiskCreds']['token']
        else:
            secret = input('Enter YaDisk app secret: ')
            token = input('Enter YaDisk app token: ')

        y = yadisk.YaDisk(id='6cbaceb74e684cfab2f28d77cdc120e0', secret=secret, token=token)

        assert y.check_token(), 'Invalid token.'

        if cache_credentials:
            config = configparser.ConfigParser()
            config['YaDiskCreds'] = {
                'secret': secret,
                'token': token
            }
            config.write(self._cred_cache_path.open('w', encoding='utf-8'))

        self._y = y
        self._logged_in = True

        logger.info('Logged in to YandexDisk')

    def _ensure_folder(self, folder: Path):
        if folder.as_posix() == 'app:':
            return
        if not self._y.exists(folder.as_posix()):
            self._ensure_folder(folder.parent)
            logger.debug(f'Creating folder {folder} on remote')
            self._y.mkdir(folder.as_posix())

    def _traverse_remote(self, remote_path: Path) -> List[Path]:
        logger.debug(f'Traversing remote folder {remote_path}')
        files = []
        for item in self._y.listdir(remote_path):
            if item.type == 'dir':
                files += self._traverse_remote(remote_path / item.name)
            else:
                files.append(remote_path / item.name)
        return files

    def _traverse_local(self, local_path: Path) -> List[Path]:
        logger.debug(f'Traversing local folder {local_path}')
        return [item for item in local_path.rglob('*') if item.is_file()]

    @logger.catch
    def download(self, remote: Union[str, Path], local: Optional[Union[str, Path]] = None):
        '''Download an object from remote to local

        Args:
            remote_path (Union[str, Path]): Path to the object in the cloud
            local_path (Optional[Union[str, Path]]): Path to the object on the machine.
            Duplicates remote if not present.
        '''
        if self._disabled:
            return

        assert self._logged_in, 'You must log in first'

        remote_path = Path('app:', remote)
        if local is None:
            local = remote
        local_path = Path(local)

        logger.info(f'Downloading {remote_path} to {local_path}')

        if self._y.is_dir(remote_path.as_posix()):
            logger.debug('Remote path is a folder, traversing')
            for item in self._traverse_remote(remote_path):
                self.download(item.relative_to('app:'), local_path / item.relative_to(remote_path))
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._y.download(remote_path.as_posix(), local_path.as_posix())

    @logger.catch
    def upload(self, local: Union[str, Path], remote: Optional[Union[str, Path]] = None):
        '''Upload an object from local to remote

        Args:
            local (Union[str, Path]): Path to the object on the machine
            remote (Optional[Union[str, Path]]): Path to the object in the cloud. Duplicates local if not present
        '''
        if self._disabled:
            return

        assert self._logged_in, 'You must log in first'

        local_path = Path(local)
        if remote is None:
            remote = local
        remote_path = Path('app:', remote)

        logger.info(f'Uploading {local_path} to {remote_path}')

        if local_path.is_dir():
            for item in self._traverse_local(local_path):
                self.upload(item, (remote_path / item.relative_to(local_path)).relative_to('app:'))
        else:
            self._ensure_folder(remote_path.parent)
            self._y.upload(local_path.as_posix(), remote_path.as_posix())
