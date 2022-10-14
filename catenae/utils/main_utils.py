"""_summary_
"""

import argparse
import logging
import logging.config

from catenae.core import extraction
from catenae.core import dsm
from catenae.core import corpus
from catenae.core import analysis

from catenae.core import statistics as stats
from catenae.utils import files_utils as futils


logger = logging.getLogger(__name__)


def compute_pos_stats(args: argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    corpus_dir = args.corpus_dirpath
    pos_list = args.pos

    stats.compute_pos_distribution(output_dir, corpus_dir, pos_list)


def compute_morph_stats(args: argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    corpus_dir = args.corpus_dirpath
    trait = args.trait
    values_list = args.values

    stats.compute_morph_distribution(output_dir, corpus_dir, trait, values_list)


# def compute_mood_stats(args):
#     output_dir = args.output_dir
#     corpus_dir = args.corpus_dirpath
#     mood_list = args.mood
#
#     stats.compute_mood_distribution(output_dir, corpus_dir, mood_list)
#
#
# def compute_form_stats(args):
#     output_dir = args.output_dir
#     corpus_dir = args.corpus_dirpath
#     form_list = args.form
#
#     stats.compute_tense_distribution(output_dir, corpus_dir, form_list)


def compute_verbedges_stats(args: argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    corpus_dir = args.corpus_dirpath
    edges = args.number_edges

    stats.compute_verbedges_distribution(output_dir, corpus_dir, edges)


def compute_sbj_stats(args: argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    corpus_dir = args.corpus_dirpath

    stats.compute_sbj_distribution(output_dir, corpus_dir)


def compute_synrel_stats(args: argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    corpus_dir = args.corpus_dirpath
    synrel_list = args.synrels

    stats.compute_synrel_distribution(output_dir, corpus_dir, synrel_list)


def extract_catenae(args: argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): _description_
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
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """

    output_dir = futils.check_or_create_dir(args.output_dir)

    items_filepath = args.items_filepath
    totals_filepath = args.totals_filepath
    catenae_filepath = args.catenae_filepath

    extraction.weight_catenae(output_dir,
                              items_filepath, totals_filepath, catenae_filepath)


def filter_catenae(args: argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): _description_
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
    """_summary_

    Args:
        args (argparse.Namespace): _description_
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
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """

    # TODO: add parameter for dimensions

    output_dir = futils.check_or_create_dir(args.output_dir)
    cooccurrences_filepath = args.cooccurrences_filepath
    frequences_filepath = args.frequences_filepath
    TOT = args.total

    dsm.build(output_dir, cooccurrences_filepath, frequences_filepath, TOT)


def sample_input(args: argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    input_dir = args.corpus_dirpath
    size = args.size
    seed = args.seed

    corpus.sample(output_dir, input_dir, size, seed)


def udparse(args: argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    input_dir = args.input_dir
    model_path = args.model

    corpus.parse(output_dir, input_dir, model_path)


def correlate(args: argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    filenames_list = args.files_list
    topk = args.top_k
    mi = args.mi_threshold
    frequency = args.frequency_threshold

    analysis.correlate(output_dir, filenames_list, topk, mi, frequency)


def corecatenae(args: argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    input_filenames_list = args.input_files_list
    babbling_filenames_list = args.babbling_files_list
    topk = args.top_k

    analysis.corecatenae(output_dir, input_filenames_list, babbling_filenames_list, topk)


def extract_sentences(args: argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    input_dir = args.input_dir
    catenae_list = args.catenae_list

    extraction.extract_sentences(output_dir, input_dir, catenae_list)


def compute_simmatrix(args: argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    input_dsm_vec = args.dsm_vec
    input_dsm_idx = args.dsm_idx
    left_subset_path = args.reduced_left_matrix
    right_subset_path = args.reduced_right_matrix
    working_memory = args.working_memory

    if args.chunked:
        dsm.compute_simmatrix_npy(output_dir, input_dsm_vec, input_dsm_idx,
                                  left_subset_path, right_subset_path,
                                  working_memory)
    else:
        dsm.compute_simmatrix(output_dir, input_dsm_vec, input_dsm_idx,
                              left_subset_path, right_subset_path)


def glass(args: argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """
    output_dir = futils.check_or_create_dir(args.output_dir)


def reduce_simmatrix(args: argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """

    output_dir = futils.check_or_create_dir(args.output_dir)
    similarities_values = args.similarities_values
    index_map = args.index_map
    top_k = args.top_k


    dsm.reduce(output_dir, similarities_values, index_map, top_k)
