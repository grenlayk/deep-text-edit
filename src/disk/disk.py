from pathlib import Path
from typing import Any, Union
import yadisk


class Disk:
    def __init__(self):
        self._logged_in = False

    def login(self):
        secret = input("Enter YaDisk app secret: ")
        token = input("Enter YaDisk app token: ")
        y = yadisk.YaDisk(id="6cbaceb74e684cfab2f28d77cdc120e0", secret=secret, token=token)

        assert y.check_token(), "Invalid token."
        self._y = y
        self._logged_in = True

    def download(self, remote_path: Union[str, Path], local_path: Union[str, Path]):
        assert self._logged_in, "You must log in first"
        self._y.download(f'app:/{remote_path}', local_path)

    def upload(self, local_path: Union[str, Path], remote_path: Union[str, Path]):
        assert self._logged_in, "You must log in first"
        self._y.upload(local_path, f'app:/{remote_path}')


disk = Disk()
