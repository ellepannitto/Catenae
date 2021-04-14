import itertools

def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"""
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def load_catenae_set(filepath):

    catenae = set()
    with open(filepath) as fin:
        fin.readline()
        for line in fin:
            line = line.strip().split("\t")
            catena, _, _ = line
            catenae.add(catena)

    return catenae
