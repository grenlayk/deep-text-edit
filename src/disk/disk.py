import configparser
from pathlib import Path
from typing import Union
from loguru import logger

import yadisk


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

    def login(self, use_cache=True, cache_credentials=True):
        '''
        Login to Yandex Disk. You must call this method before you can use other functions.
        '''

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
        if not self._y.exists(folder):
            self._ensure_folder(f'app:/{folder.parent}')
            self._y.mkdir(f'app:/{folder}')

    def download(self, remote_path: Union[str, Path], local_path: Union[str, Path]):
        '''Download an object from remote_path to local_path

        Args:
            remote_path (Union[str, Path]): Path to the object in the cloud
            local_path (Union[str, Path]): Path to the object on the machine
        '''

        assert self._logged_in, 'You must log in first'
        self._y.download(f'app:/{remote_path}', str(local_path))

    def upload(self, local_path: Union[str, Path], remote_path: Union[str, Path]):
        '''Upload an object from local_path to remote_path

        Args:
            local_path (Union[str, Path]): Path to the object on the machine
            remote_path (Union[str, Path]): Path to the object in the cloud
        '''

        assert self._logged_in, 'You must log in first'
        if isinstance(remote_path, str):
            remote_path = Path(remote_path)
        self._ensure_folder(remote_path)
        self._y.upload(str(local_path), f'app:/{remote_path}')
