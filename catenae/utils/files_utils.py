import os
import glob
import logging

from typing import List


logger = logging.getLogger(__name__)


def get_filenames(input_dir: str) -> List[str]:

    if os.path.isdir(input_dir):
        return glob.glob(input_dir+"/*", recursive=True)
    else:
        return [input_dir]


def check_or_create_dir(path: str) -> None:
    """_summary_

    Args:
        path (str): _description_
    """

    if not os.path.exists(path):
        logger.info(f"Creating folder {path}")
        os.mkdir(path)
    else:
        logger.info(f"Checked path: {path}")

    return path
