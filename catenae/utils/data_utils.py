import itertools

def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"""
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def load_catenae_set(filepath, topk):

    catenae = set()
    with open(filepath) as fin:
        fin.readline()
        for lineno, line in enumerate(fin):
            if lineno < topk:
                line = line.strip().split("\t")
                catena, _, _ = line
                catenae.add(catena)
            else:
                break

    return catenae
