"""
Set of utilities used to load and manipulated different kind of data used in the package.
"""
import itertools
import gzip

from typing import Set, Any, Iterable, Tuple
from pathlib import Path

import tqdm
import numpy as np

from catenae.utils import files_utils as futils

import filemerger.utils as fmergerutils
# from FileMerger.filesmerger import utils as fmergerutils


def grouper(iterable: Iterable[Any], n: int, fillvalue: Any = None) -> Iterable: # pylint:disable=C0103
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


def load_catenae_set(filepath: Path, topk: int, catenae_set: Set[Any] = None) -> Set[Any]:
    """Load first topk catenae into a set.

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

    with fmergerutils.open_file_by_extension(futils.get_str_path(filepath)) as fin:
        fin.readline()
        for lineno, line in enumerate(fin):
            line = line.strip().split("\t")
            catena, *_ = line

            if lineno < topk:
                catenae.add(catena)
            else:
                break

    return catenae


def _generate_lines(input_path_vec: Path,
                    vectors_to_load: set = None) ->  Iterable[str]:
    """_summary_

    Args:
        input_path_vec (Path): _description_
        vectors_to_load (set, optional): _description_. Defaults to None.

    Returns:
        Iterable[str]: _description_

    Yields:
        Iterator[Iterable[str]]: _description_
    """

    if len(vectors_to_load) == 0:
        return gzip.open(input_path_vec, "rt")

    with gzip.open(input_path_vec, "rt") as fin_vec:

        for line_no, vec_line in tqdm.tqdm(enumerate(fin_vec),
                                           desc=f"Reading file {input_path_vec}"):

            if line_no in vectors_to_load:
                yield vec_line


def load_vectors(input_path_vec: Path,
                 vectors_to_load: set = None) -> np.ndarray:
    """_summary_

    Args:
        input_path_vec (Path): _description_
        input_path_idx (Path): _description_
        vectors_to_load (set, optional): _description_. Defaults to None.

    Returns:
        np.ndarray: _description_
    """

    vectors = np.loadtxt(_generate_lines(input_path_vec, vectors_to_load),
                         dtype=np.float32)

    return vectors

def remap(full_set, set1, set2):
    full_set = sorted(full_set)

    new_set1 = []
    new_set2 = []

    for i, el_id in enumerate(full_set):
        if el_id in set1:
            new_set1.append(i)
        if el_id in set2:
            new_set2.append(i)

    return new_set1, new_set2


def selectrows(matrix: np.ndarray, idxs: Set[int]) -> np.ndarray:
    return matrix[sorted(idxs), :]


class DefaultList(list):
    """Extend class List.

    The class makes it possible to assign a value is a out of bound position, the list will be
    filled with default values up to that position.
    """

    def __init__(self, iterable: Iterable[Any], default_value: Any = None):
        """Initialize list from iterable. Store default value.

        Args:
            iterable (Iterable[Any]): iterable to build the list from
            default_value (Any, optional): Default value used to fill the list. Defaults to None.
        """
        self.default_value = default_value
        super().__init__(iterable)

    def __setitem__(self, index: int, item: Any):
        """Insert item in position index. If position index is larger that the length of the list,
        the list is filled with default values.

        Args:
            index (int): position where the item will be placed
            item (Any): item to place
        """

        self.fill(index)

        super().append(index, item)

    def fill(self, index: int):
        """Fills the list with default value from position len to position index-1

        Args:
            index (int): _description_
        """

        while index > len(self):
            super().append(self.default_value)


def dump_idxs(output_filepath: Path, full_dsm_idx: Path, vectors_set: set) -> None:
    """_summary_

    Args:
        output_dir (Path): _description_
        full_dsm_idx (Path): _description_
        vectors_set (set): _description_
    """

    with open(output_filepath, "w") as fout, \
        gzip.open(full_dsm_idx, "rt") as fin_idxs:

        for idx_line in tqdm.tqdm(fin_idxs):
            idx_line = idx_line.strip()
            if idx_line in vectors_set:
                print(idx_line, file=fout)


def load_map(input_fpath: Path) -> Tuple[dict, dict]:

    idx_to_cat = {}
    cat_to_idx = {}

    with gzip.open(input_fpath, "rt") as fin:
        for i, line in tqdm.tqdm(enumerate(fin), desc="Loading catenae to idxs dict"):
            catena = line.strip()
            idx_to_cat[i] = catena
            cat_to_idx[catena] = i

    return idx_to_cat, cat_to_idx


if __name__ == "__main__":
    lst = DefaultList([1,2,3], "_")
    print(type(lst))
    # type(lst)
    lst[5] = "a"

    print(lst)
