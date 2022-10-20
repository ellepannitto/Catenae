"""
Set of utilities for interacting with files and directories.
"""
import os
import glob
import logging

from pathlib import Path

from typing import Iterable, List, Tuple, TextIO


logger = logging.getLogger(__name__)


def get_filenames(input_dir: str) -> Iterable[str]:
    """Returns paths to files contained in directory.

    Args:
        input_dir (str): List of paths to directories.

    Returns:
        Iterable[str]: List of paths to files contained in directories.
    """

    if os.path.isdir(input_dir):
        return glob.glob(input_dir+"/*", recursive=True)

    return [input_dir]


def check_or_create_dir(path: str) -> Path:
    """Check if a directory exists already. If not, create the directory.

    Args:
        path (str): path to directory.

    Returns:
        Path: Path object to directory (either existent or newly created).
    """

    path = Path(path)

    if not path.exists():
        logger.info("Creating folder %s", path)
        path.mkdir(parents=True)
    else:
        logger.info("Checked path: %s", path)

    return path


def print_formatted(list_of_tuples: List[Tuple], file_handler: TextIO, sep: str = "\t") -> None:
    """_summary_

    Args:
        list_of_tuples (List[Tuple]): _description_
        file_handler (TextIO): _description_
        sep (str, optional): _description_. Defaults to "\t".
    """

    for tup in list_of_tuples:
        print(sep.join(tup), file=file_handler)
