#!/bin/env python3
# pylint: disable=no-value-for-parameter

from importlib import import_module
from pathlib import Path

import click
from loguru import logger


@click.command()
@click.argument('config_file',
                type=click.Path(exists=True, dir_okay=False, readable=True),
                default='')
@logger.catch
def run(config_file):
    config_file = Path(config_file)
    config_file = (config_file.parent / config_file.stem).as_posix().replace('/', '.')
    imported = import_module(config_file)
    config = imported.Config()
    config.run()


if __name__ == '__main__':
    run()
