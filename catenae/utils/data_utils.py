import itertools
import gzip
import tqdm

from typing import Set, Any, Iterable, Tuple, Generator

import numpy as np

from FileMerger.filesmerger import utils as fmergerutils

def grouper(iterable: Iterable[Any], n: int, fillvalue: Any = None) -> Iterable:
    """Collect data into fixed-length chunks or blocks.

    grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx

    Args:
        iterable (Iterable[Any]): _description_
        n (int): _description_
        fillvalue (Any, optional): _description_. Defaults to None.

    Returns:
        Iterable: _description_
    """

    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def load_catenae_set(filepath: str, topk: int, catenae_set: Set = None) -> Set:
    """Loads first topk catenae into a set.

    Args:
        filepath (str): _description_
        topk (int): _description_
        catenae_set (Set, optional): _description_. Defaults to None.

    Returns:
        Set: _description_
    """

    if catenae_set is None:
        catenae = set()
    else:
        catenae = catenae_set

    with fmergerutils.open_file_by_extension(filepath) as fin:
        fin.readline()
        for lineno, line in enumerate(fin):
            line = line.strip().split("\t")
            catena, *_ = line

            if lineno < topk:
                catenae.add(catena)
            else:
                break

    return catenae


def _generate_lines(input_path_vec:str, input_path_idx: str, 
                    vectors_to_load: set = None) ->  Iterable[str]:
    
    with gzip.open(input_path_vec, "rt") as fin_vec, \
        gzip.open(input_path_idx, "rt") as fin_idx:

        for idx_line in tqdm.tqdm(fin_idx, desc=f"Reading file {input_path_vec}"):
            vec_line = fin_vec.readline()
            idx_line = " ".join(idx_line.strip().split("|")) #TODO: change

            if not vectors_to_load or idx_line in vectors_to_load:
                yield vec_line


def load_vectors(input_path_vec: str, input_path_idx: str, 
                 vectors_to_load: set = None) -> np.ndarray:

    vectors = np.loadtxt(_generate_lines(input_path_vec, input_path_idx, vectors_to_load),
                         dtype=np.float32)

    return vectors
