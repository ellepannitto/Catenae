import logging
import collections
import gzip
import math

import numpy as np
import scipy as sp
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances_chunked
from scipy.sparse.linalg import svds, eigs

from catenae.utils import data_utils as dutils

from FileMerger.filesmerger import utils as fmergerutils


logger = logging.getLogger(__name__)


def build(output_dir, coocc_filepath, freqs_filepath, TOT, svd_dim = 300):

    item_to_id = {}
    id_max = 0
    matrix = collections.defaultdict(lambda: {})

    with fmergerutils.open_file_by_extension(coocc_filepath) as fin_cocc, \
         fmergerutils.open_file_by_extension(freqs_filepath) as fin_freqs_left, \
         fmergerutils.open_file_by_extension(freqs_filepath) as fin_freqs_right, \
         gzip.open(output_dir + "catenae-ppmi.gz", "wt") as fout:

        lineno = 1

        line_cocc = fin_cocc.readline()

        line_freq_left = fin_freqs_left.readline()

        cat_l, freq_l = line_freq_left.strip().split("\t")
        freq_l = float(freq_l)

        line_freq_right = fin_freqs_right.readline()
        cat_r, freq_r = line_freq_right.strip().split("\t")
        freq_r = float(freq_r)


        while line_cocc:

            cats, freq = line_cocc.strip().split("\t")
            cat1, cat2 = cats.split(" ")
            freq = float(freq)

            while cat_l < cat1:
                line_freq_left = fin_freqs_left.readline()
                cat_l, freq_l, = line_freq_left.strip().split("\t")
                freq_l = float(freq_l)

            if cat_r > cat2:
                fin_freqs_right = fmergerutils.open_file_by_extension(freqs_filepath)
                line_freq_right = fin_freqs_right.readline()
                cat_r, freq_r = line_freq_right.strip().split("\t")
                freq_r = float(freq_r)

            while cat_r < cat2:
                line_freq_right = fin_freqs_right.readline()
                cat_r, freq_r = line_freq_right.strip().split("\t")
                freq_r = float(freq_r)

            assert cat1 == cat_l, "MISSING CATENA"
            assert cat2 == cat_r, "MISSING CATENA"

            ppmi = freq * math.log(freq*TOT/(freq_l*freq_r))
            if ppmi > 0:
                print("{}\t{}\t{}\t{}".format(cat1, cat2, freq, ppmi), file=fout)

            if not cat1 in item_to_id:
                item_to_id[cat1] = id_max
                id_max += 1

            if not cat2 in item_to_id:
                item_to_id[cat2] = id_max
                id_max += 1

            matrix[item_to_id[cat1]][item_to_id[cat2]] = ppmi
            matrix[item_to_id[cat2]][item_to_id[cat1]] = ppmi

            line_cocc = fin_cocc.readline()
            lineno += 1

            if not lineno % 10000:
                logger.info(f"PROCESSING LINE {lineno}")

    id_to_item = [0]*id_max
    for item, id in item_to_id.items():
        id_to_item[id] = item

    S = sp.sparse.dok_matrix((id_max, id_max), dtype=np.float32)

    for el1 in matrix:
        for el2 in matrix[el1]:
            S[el1, el2] = matrix[el1][el2]

    S = sp.sparse.csc_matrix(S)
    u, s, vt = svds(S, k=svd_dim)

    with gzip.open(output_dir + "catenae-dsm.gz", "wt") as fout:
        el_no = 0
        for el in u:
            print("{}\t{}".format(id_to_item[el_no], " ".join(str(x) for x in el)), file=fout)
            el_no += 1


def compute_simmatrix(output_dir: str, input_dsm_vec: str, input_dsm_idx: str, 
                      left_subset_path: str, right_subset_path: str) -> None:
    """_summary_

    Args:
        output_dir (str): _description_
        input_dsm (str): _description_
        left_subset_path (str): _description_
        right_subset_path (str): _description_
    """

    left_vectors_to_load = set()
    if not left_subset_path == "all":
        left_vectors_to_load = dutils.load_catenae_set(left_subset_path, math.inf)

    right_vectors_to_load = set()
    if not right_subset_path == "all":
        right_vectors_to_load = dutils.load_catenae_set(right_subset_path, math.inf)
    
    vectors_to_load = None
    if len(left_vectors_to_load) and len(right_vectors_to_load):
        vectors_to_load = left_vectors_to_load.union(right_vectors_to_load)

    print(vectors_to_load)

    DSM = dutils.load_vectors(input_dsm_vec, input_dsm_idx, vectors_to_load)

    simmatrix = pairwise_distances_chunked(DSM, metric="cosine", working_memory=16000)
    # simmatrix = pairwise_distances_chunked(DSM, metric="cosine")

    for chunk in simmatrix:
        print(chunk.shape)
        input()