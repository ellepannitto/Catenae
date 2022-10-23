"""
Set of utilities for interacting with files and directories.
"""
import os
import glob
import logging
import sys

from pathlib import Path

from typing import Iterable, List, Tuple, TextIO


logger = logging.getLogger(__name__)


def check_path(path: str) -> Path:
    """Check if a path exist

    Args:
        path (str): string passed as input from user

    Returns:
        Path: Path object pointing to resource, if existent
    """

    path_obj = Path(path)

    if not path_obj.exists():
        sys.exit(f'Path {path} does not exist')

    return path_obj


def get_filenames(input_dir: str) -> Iterable[str]:
    # TODO: parametrize regex to look for
    # TODO: use in statistics instead of if.
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
