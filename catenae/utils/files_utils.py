"""_summary_

Returns:
    _type_: _description_
"""
import os
import glob
import logging

from pathlib import Path

from typing import List


logger = logging.getLogger(__name__)


def get_filenames(input_dir: str) -> List[str]:
    """_summary_

    Args:
        input_dir (str): _description_

    Returns:
        List[str]: _description_
    """

    if os.path.isdir(input_dir):
        return glob.glob(input_dir+"/*", recursive=True)

    return [input_dir]


def check_or_create_dir(path: str) -> Path:
    """_summary_

    Args:
        path (str): _description_

    Returns:
        Path: _description_
    """
    
    p = Path(path)

    if not p.exists():
        logger.info("Creating folder %s", path)
        p.mkdir(parents=True)
    else:
        logger.info("Checked path: %s", path)

    return p
