#!/bin/env python3
# pylint: disable=no-value-for-parameter

from importlib import import_module
from pathlib import Path
from src.disk import disk

import click
from loguru import logger


@click.command()
@click.argument('config_file',
                type=click.Path(exists=True, dir_okay=False, readable=True),
                default='./src/config/color.py')
@click.option('--yadisk', '--enable_disk',
                is_flag=True, show_default=True, default=False, help='Enable uploading checkpoints to Yandex.Disk')
@logger.catch
def run(config_file, yadisk):
    config_file = Path(config_file)
    config_file = (config_file.parent / config_file.stem).as_posix().replace('/', '.')
    imported = import_module(config_file)
    disk.set_disabled(not yadisk)
    config = imported.Config()
    config.run()


if __name__ == '__main__':
    run()
