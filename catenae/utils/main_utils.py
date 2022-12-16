"""
Functions to dispatch parameters to functions.
"""

import sys
import argparse
import logging
import logging.config

from catenae.core import extraction
from catenae.core import dsm
from catenae.core import corpus
from catenae.core import analysis
from catenae.core import glassify
from catenae.core import statistics

from catenae.utils import files_utils as futils


logger = logging.getLogger(__name__)


def compute_pos_stats(args: argparse.Namespace) -> None:
    """Dispatch parameters for Part-of-Speech statistics.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    corpus_dir = futils.check_path(args.corpus_dirpath)
    pos_list = args.pos

    statistics.compute_pos_distribution(output_dir, corpus_dir, pos_list)


def compute_morph_stats(args: argparse.Namespace) -> None:
    """Dispatch parameters for morphology statistics.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    corpus_dir = futils.check_path(args.corpus_dirpath)
    trait = args.trait
    values_list = args.values

    statistics.compute_morph_distribution(output_dir, corpus_dir, trait, values_list)


def compute_verbedges_stats(args: argparse.Namespace) -> None:
    """Dispatch parameters for statistics on verb edges.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    corpus_dir = futils.check_path(args.corpus_dirpath)
    edges = args.number_edges

    statistics.compute_verbedges_distribution(output_dir, corpus_dir, edges)


def compute_sbj_stats(args: argparse.Namespace) -> None:
    """Dispatch parameters for statistics on distribution of subjects.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    corpus_dir = futils.check_path(args.corpus_dirpath)

    statistics.compute_sbj_distribution(output_dir, corpus_dir)


def compute_synrel_stats(args: argparse.Namespace) -> None:
    """Dispatch parameters for statistics on syntactic relations.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    corpus_dir = futils.check_path(args.corpus_dirpath)
    synrel_list = args.synrels

    statistics.compute_synrel_distribution(output_dir, corpus_dir, synrel_list)


def extract_catenae(args: argparse.Namespace) -> None:
    """Dispatch parameters for extracting catenae.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    if len(list(output_dir.iterdir())) > 0:
        sys.exit(f"ERROR: directory {output_dir} is not empty!")

    corpus_dir = futils.check_path(args.corpus_dirpath)

    min_len_sentence = args.min_len_sentence
    max_len_sentence = args.max_len_sentence
    sentences_batch_size = args.sentences_batch_size
    min_freq = args.frequency_threshold

    min_len_catena = args.min_len_catena
    max_len_catena = args.max_len_catena

    extraction.extract_catenae(output_dir, corpus_dir,
                               min_len_sentence, max_len_sentence, sentences_batch_size,
                               min_freq, min_len_catena, max_len_catena)


def weight_catenae(args: argparse.Namespace) -> None:
    """Dispatch parameters for weighting catenae.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)

    items_filepath = futils.check_path(args.items_filepath)
    totals_filepath = futils.check_path(args.totals_filepath)
    catenae_filepath = futils.check_path(args.catenae_filepath)

    extraction.weight_catenae(output_dir,
                              items_filepath, totals_filepath, catenae_filepath)


def filter_catenae(args: argparse.Namespace) -> None:
    """Dispatch parameters for filtering catenae.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)

    input_file = futils.check_path(args.input_filepath)
    frequency_threshold = args.min_freq
    weight_threshold = args.min_weight
    min_len_catena = args.min_len_catena
    max_len_catena = args.max_len_catena

    extraction.filter_catenae(output_dir, input_file, frequency_threshold, weight_threshold,
                              min_len_catena, max_len_catena)


def extract_cooccurrences(args: argparse.Namespace) -> None:
    """Dispatch parameters for extracting co-occurrences.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    if len(list(output_dir.iterdir())) > 0:
        sys.exit(f"ERROR: directory {output_dir} is not empty!")


    corpus_dir = futils.check_path(args.corpus_dirpath)
    accepted_catenae_filepath = futils.check_path(args.accepted_catenae)
    top_k = args.topk

    min_len_sentence = args.min_len_sentence
    max_len_sentence = args.max_len_sentence
    sentences_batch_size = args.sentences_batch_size
    min_freq = args.frequency_threshold

    min_len_catena = args.min_len_catena
    max_len_catena = args.max_len_catena

    include_words = args.include_len_one_items
    words_filepath = futils.check_path(args.words_filepath)

    extraction.extract_coccurrences(output_dir, corpus_dir, accepted_catenae_filepath, top_k,
                                    min_len_sentence, max_len_sentence, sentences_batch_size,
                                    min_freq, min_len_catena, max_len_catena,
                                    include_words, words_filepath)


def build_dsm(args: argparse.Namespace) -> None:
    """Dispatch parameters for building distributional semantic model.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    # TODO: add parameter for dimensions

    output_dir = futils.check_or_create_dir(args.output_dir)
    cooccurrences_filepath = futils.check_path(args.cooccurrences_filepath)
    frequences_filepath = futils.check_path(args.frequences_filepath)
    tot_frequency = args.total

    dsm.build(output_dir, cooccurrences_filepath, frequences_filepath, tot_frequency)


def sample_input(args: argparse.Namespace) -> None:
    """Dispatch parameters for sampling input data.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    input_dir = futils.check_path(args.corpus_dirpath)
    size = args.size
    seed = args.seed

    corpus.sample(output_dir, input_dir, size, seed)


def udparse(args: argparse.Namespace) -> None:
    """Dispatch parameters for parsing input data.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    input_dir = futils.check_path(args.input_dir)
    model_path = futils.check_path(args.model)

    corpus.parse(output_dir, input_dir, model_path)


def correlate(args: argparse.Namespace) -> None:
    """Dispatch parameters for computing Spearman correlation.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    filenames_list = [futils.check_path(fname) for fname in args.files_list]
    topk = args.top_k
    mutual_information = args.mi_threshold
    frequency = args.frequency_threshold

    analysis.correlate(output_dir, filenames_list, topk, mutual_information, frequency)


def corecatenae(args: argparse.Namespace) -> None:
    """Dispatch parameters for extracting core catenae.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)

    input_filenames_list = []
    for filename in args.input_files_list:
        input_filenames_list.append(futils.check_path(filename))

    babbling_filenames_list = []
    for filename in args.babbling_files_list:
        babbling_filenames_list.append(futils.check_path(filename))
    topk = args.top_k

    analysis.corecatenae(output_dir, input_filenames_list, babbling_filenames_list, topk)


def extract_sentences(args: argparse.Namespace) -> None:
    """Dispatch parameters for extracting sentences containing specific catenae.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    input_dir = futils.check_path(args.input_dir)
    catenae_list = futils.check_path(args.catenae_list)

    extraction.extract_sentences(output_dir, input_dir, catenae_list)


def compute_simmatrix(args: argparse.Namespace) -> None:
    """Dispatch parameters for computing matrix of similarities between vectors.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    input_dsm_vec = futils.check_path(args.dsm_vec)
    input_dsm_idx = futils.check_path(args.dsm_idx)
    left_subset_path = futils.check_path(args.reduced_left_matrix)
    right_subset_path = futils.check_path(args.reduced_right_matrix)
    working_memory = args.working_memory

    if args.chunked:
        dsm.compute_simmatrix_chunked(output_dir, input_dsm_vec, input_dsm_idx,
                                      left_subset_path, right_subset_path,
                                      working_memory)
    else:
        dsm.compute_simmatrix(output_dir, input_dsm_vec, input_dsm_idx,
                              left_subset_path, right_subset_path)


def query_neighbors(args: argparse.Namespace) -> None:
    """Dispatch parameters for querying nearest neighbors.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """
    input_dsm_sim = futils.check_path(args.similarities_matrix)
    input_dsm_idx = futils.check_path(args.indexes_matrix)
    input_idx_map = futils.check_path(args.indexes_map)

    dsm.query_neighbors(input_dsm_sim, input_dsm_idx, input_idx_map)


def reduce_simmatrix(args: argparse.Namespace) -> None:
    """Dispatch parameters for reduding matrix of similarities.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    similarities_values = []
    for filename in args.similarities_values:
        similarities_values.append(futils.check_path(filename))
    top_k = args.top_k

    dsm.reduce(output_dir, similarities_values, top_k)


def glassify_matrix(args: argparse.Namespace) -> None:
    """Dispatch parameters for projecting catenae on data.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """
    output_dir = futils.check_or_create_dir(args.output_dir)
    input_dir = futils.check_path(args.input_dir)
    catenae_path = futils.check_path(args.catenae)
    min_len_catena = args.min_len_catena
    max_len_catena = args.max_len_catena
    multiprocess = args.multiprocess
    n_workers = args.n_workers

    glassify.compute_matrix(output_dir, input_dir, catenae_path,
                            min_len_catena, max_len_catena,
                            multiprocess, n_workers)


def sentence_matrix(args:argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    input_dir = futils.check_path(args.input_dir)
    catenae_fpath = futils.check_path(args.catenae)
    multiprocess = args.multiprocessing
    n_workers = args.n_workers

    glassify.sentence_matrix(output_dir, input_dir, catenae_fpath,
                             multiprocess, n_workers)


def glassify_collapse(args: argparse.Namespace) -> None:
    """Dispatch parameters for projecting catenae on data.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """
    output_dir = futils.check_or_create_dir(args.output_dir)
    input_dir = futils.check_path(args.input_dir)
    catenae_fpath = futils.check_path(args.catenae)
    multiprocess = args.multiprocess
    n_workers = args.n_workers
    chunksize = args.chunksize

    glassify.collapse_matrix(output_dir, input_dir, catenae_fpath,
                             multiprocess, n_workers, chunksize)


#TODO: fix docstrings