"""
Functions to dispatch parameters to functions.
"""

import argparse
import logging
import logging.config

from catenae.core import extraction
from catenae.core import dsm
from catenae.core import corpus
from catenae.core import analysis
from catenae.core import glassify

from catenae.core import statistics as stats
from catenae.utils import files_utils as futils


logger = logging.getLogger(__name__)


def compute_pos_stats(args: argparse.Namespace) -> None:
    """Dispatch parameters for Part-of-Speech statistics.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    corpus_dir = args.corpus_dirpath
    pos_list = args.pos

    stats.compute_pos_distribution(output_dir, corpus_dir, pos_list)


def compute_morph_stats(args: argparse.Namespace) -> None:
    """Dispatch parameters for morphology statistics.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    corpus_dir = args.corpus_dirpath
    trait = args.trait
    values_list = args.values

    stats.compute_morph_distribution(output_dir, corpus_dir, trait, values_list)


def compute_verbedges_stats(args: argparse.Namespace) -> None:
    """Dispatch parameters for statistics on verb edges.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    corpus_dir = args.corpus_dirpath
    edges = args.number_edges

    stats.compute_verbedges_distribution(output_dir, corpus_dir, edges)


def compute_sbj_stats(args: argparse.Namespace) -> None:
    """Dispatch parameters for statistics on distribution of subjects.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    corpus_dir = args.corpus_dirpath

    stats.compute_sbj_distribution(output_dir, corpus_dir)


def compute_synrel_stats(args: argparse.Namespace) -> None:
    """Dispatch parameters for statistics on syntactic relations.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    corpus_dir = args.corpus_dirpath
    synrel_list = args.synrels

    stats.compute_synrel_distribution(output_dir, corpus_dir, synrel_list)


def extract_catenae(args: argparse.Namespace) -> None:
    """Dispatch parameters for extracting catenae.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    corpus_dir = args.corpus_dirpath

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

    items_filepath = args.items_filepath
    totals_filepath = args.totals_filepath
    catenae_filepath = args.catenae_filepath

    extraction.weight_catenae(output_dir,
                              items_filepath, totals_filepath, catenae_filepath)


def filter_catenae(args: argparse.Namespace) -> None:
    """Dispatch parameters for filtering catenae.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)

    input_file = args.input_filepath
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
    corpus_dir = args.corpus_dirpath
    accepted_catenae_filepath = args.accepted_catenae
    top_k = args.topk

    min_len_sentence = args.min_len_sentence
    max_len_sentence = args.max_len_sentence
    sentences_batch_size = args.sentences_batch_size
    min_freq = args.frequency_threshold

    min_len_catena = args.min_len_catena
    max_len_catena = args.max_len_catena

    include_words = args.include_len_one_items
    words_filepath = args.words_filepath

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
    cooccurrences_filepath = args.cooccurrences_filepath
    frequences_filepath = args.frequences_filepath
    tot_frequency = args.total

    dsm.build(output_dir, cooccurrences_filepath, frequences_filepath, tot_frequency)


def sample_input(args: argparse.Namespace) -> None:
    """Dispatch parameters for sampling input data.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    input_dir = args.corpus_dirpath
    size = args.size
    seed = args.seed

    corpus.sample(output_dir, input_dir, size, seed)


def udparse(args: argparse.Namespace) -> None:
    """Dispatch parameters for parsing input data.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    input_dir = args.input_dir
    model_path = args.model

    corpus.parse(output_dir, input_dir, model_path)


def correlate(args: argparse.Namespace) -> None:
    """Dispatch parameters for computing Spearman correlation.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    filenames_list = args.files_list
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
    input_filenames_list = args.input_files_list
    babbling_filenames_list = args.babbling_files_list
    topk = args.top_k

    analysis.corecatenae(output_dir, input_filenames_list, babbling_filenames_list, topk)


def extract_sentences(args: argparse.Namespace) -> None:
    """Dispatch parameters for extracting sentences containing specific catenae.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    input_dir = args.input_dir
    catenae_list = args.catenae_list

    extraction.extract_sentences(output_dir, input_dir, catenae_list)


def compute_simmatrix(args: argparse.Namespace) -> None:
    """Dispatch parameters for computing matrix of similarities between vectors.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    input_dsm_vec = args.dsm_vec
    input_dsm_idx = args.dsm_idx
    left_subset_path = args.reduced_left_matrix
    right_subset_path = args.reduced_right_matrix
    working_memory = args.working_memory
    top_k = args.top_k

    if args.chunked and args.reduced:
        dsm.compute_simmatrix_and_reduce_chunked(output_dir, input_dsm_vec, input_dsm_idx,
                                                 left_subset_path, right_subset_path,
                                                 working_memory, top_k)
    elif args.chunked:
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
    input_dsm_sim = args.similarities_matrix
    input_dsm_idx = args.indexes_matrix
    input_idx_map = args.indexes_map

    dsm.query_neighbors(input_dsm_sim, input_dsm_idx, input_idx_map)


def reduce_simmatrix(args: argparse.Namespace) -> None:
    """Dispatch parameters for reduding matrix of similarities.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    similarities_values = args.similarities_values
    top_k = args.top_k

    dsm.reduce(output_dir, similarities_values, top_k)


def glassify_matrix(args: argparse.Namespace) -> None:
    """Dispatch parameters for projecting catenae on data.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """
    output_dir = futils.check_or_create_dir(args.output_dir)
    input_filename = args.input
    catenae_path = args.catenae

    glassify.compute_matrix(output_dir, input_filename, catenae_path,
                            min_len_catena=1, max_len_catena=5)


def glassify_collapse(args: argparse.Namespace) -> None:
    """Dispatch parameters for projecting catenae on data.

    Args:
        args (argparse.Namespace): Object for storing attributes provided as parameters.
    """
    output_dir = futils.check_or_create_dir(args.output_dir)
    input_dir = args.input_dir

    glassify.collapse_matrix(output_dir, input_dir)