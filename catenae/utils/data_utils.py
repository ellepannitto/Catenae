import itertools

from FileMerger.filesmerger import utils as fmergerutils

def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"""
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def load_catenae_set(filepath, topk, catenae_set=None):
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
