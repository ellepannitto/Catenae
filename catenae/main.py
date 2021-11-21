import argparse
from argparse import RawTextHelpFormatter, ArgumentDefaultsHelpFormatter
import logging.config
import os

from catenae.utils import config_utils as cutils
from catenae.core import extraction as extraction
from catenae.core import statistics as stats
from catenae.core import dsm as dsm

config_dict = cutils.load(os.path.join(os.path.dirname(__file__), 'logging_utils', 'logging.yml'))
logging.config.dictConfig(config_dict)

logger = logging.getLogger(__name__)


def _compute_pos_stats(args):
    output_dir = args.output_dir
    corpus_dir = args.corpus_dirpath
    pos_list = args.pos

    stats.compute_pos_distribution(output_dir, corpus_dir, pos_list)


def _compute_tense_stats(args):
    output_dir = args.output_dir
    corpus_dir = args.corpus_dirpath
    tense_list = args.tense

    stats.compute_tense_distribution(output_dir, corpus_dir, tense_list)

def _compute_mood_stats(args):
    output_dir = args.output_dir
    corpus_dir = args.corpus_dirpath
    mood_list = args.mood

    stats.compute_mood_distribution(output_dir, corpus_dir, mood_list)

def _compute_form_stats(args):
    output_dir = args.output_dir
    corpus_dir = args.corpus_dirpath
    form_list = args.form

    stats.compute_tense_distribution(output_dir, corpus_dir, form_list)


def _extract_catenae(args):
    output_dir = args.output_dir
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


def _weight_catenae(args):
    output_dir = args.output_dir

    items_filepath = args.items_filepath
    totals_filepath = args.totals_filepath
    catenae_filepath = args.catenae_filepath

    extraction.weight_catenae(output_dir,
                              items_filepath, totals_filepath, catenae_filepath)


def _filter_catenae(args):
    output_dir = args.output_dir

    input_file = args.input_filepath
    frequency_threshold = args.min_freq
    weight_threshold = args.min_weight
    min_len_catena = args.min_len_catena
    max_len_catena = args.max_len_catena

    extraction.filter_catenae(output_dir, input_file, frequency_threshold, weight_threshold,
                              min_len_catena, max_len_catena)


def _extract_cooccurrences(args):
    output_dir = args.output_dir
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


def _build_dsm(args):

    # TODO: add parameter for dimensions

    output_dir = args.output_dir
    cooccurrences_filepath = args.cooccurrences_filepath
    frequences_filepath = args.frequences_filepath
    TOT = args.total

    dsm.build(output_dir, cooccurrences_filepath, frequences_filepath, TOT)


def main():
    root_parser = argparse.ArgumentParser(prog='catenae', formatter_class=RawTextHelpFormatter)
    subparsers = root_parser.add_subparsers(title="actions", dest="actions")

    # Compute stats on corpus
    parser_pos_stats = subparsers.add_parser('pos-stats',
                                          description='compute list of stats for a corpus',
                                          help='compute list of stats for a corpus',
                                          formatter_class=ArgumentDefaultsHelpFormatter)
    parser_pos_stats.add_argument("-o", "--output-dir", default="data/stats/",
                              help="path to output dir")
    parser_pos_stats.add_argument("-c", "--corpus-dirpath", default="data/corpus/",
                                help="path to corpus directory")
    parser_pos_stats.add_argument("-p", "--pos", required=True, nargs="+",
                              help="Universal Part of Speech tag")
    parser_pos_stats.set_defaults(func=_compute_pos_stats)

    parser_tense_stats = subparsers.add_parser('tense-stats',
                                          description='compute list of stats for a corpus',
                                          help='compute list of stats for a corpus',
                                          formatter_class=ArgumentDefaultsHelpFormatter)
    parser_tense_stats.add_argument("-o", "--output-dir", default="data/stats/",
                              help="path to output dir")
    parser_tense_stats.add_argument("-c", "--corpus-dirpath", default="data/corpus/",
                                help="path to corpus directory")
    parser_tense_stats.add_argument("-t", "--tense", required=True, nargs="+",
                              help="Universal Part of Speech tense value")
    parser_tense_stats.set_defaults(func=_compute_tense_stats)

    parser_mood_stats = subparsers.add_parser('mood-stats',
                                          description='compute list of stats for a corpus',
                                          help='compute list of stats for a corpus',
                                          formatter_class=ArgumentDefaultsHelpFormatter)
    parser_mood_stats.add_argument("-o", "--output-dir", default="data/stats/",
                              help="path to output dir")
    parser_mood_stats.add_argument("-c", "--corpus-dirpath", default="data/corpus/",
                                help="path to corpus directory")
    parser_mood_stats.add_argument("-m", "--mood", required=True, nargs="+",
                              help="Universal Part of Speech mood value")
    parser_mood_stats.set_defaults(func=_compute_mood_stats)

    parser_form_stats = subparsers.add_parser('verbform-stats',
                                          description='compute list of stats for a corpus',
                                          help='compute list of stats for a corpus',
                                          formatter_class=ArgumentDefaultsHelpFormatter)
    parser_form_stats.add_argument("-o", "--output-dir", default="data/stats/",
                              help="path to output dir")
    parser_form_stats.add_argument("-c", "--corpus-dirpath", default="data/corpus/",
                                help="path to corpus directory")
    parser_form_stats.add_argument("-f", "--form", required=True, nargs="+",
                              help="Universal Part of Speech VerbForm value")
    parser_form_stats.set_defaults(func=_compute_form_stats)

    # Extraction of catenae
    parser_extract = subparsers.add_parser('extract',
                                           description='extract list of catenae',
                                           help='extract list of catenae',
                                           formatter_class=ArgumentDefaultsHelpFormatter)
    parser_extract.add_argument("-o", "--output-dir", default="data/results/",
                                help="path to output dir")
    parser_extract.add_argument("-c", "--corpus-dirpath", default="data/corpus/",
                                help="path to corpus directory")
    parser_extract.add_argument("-m", "--min-len-sentence", type=int, default=1,
                                help="minimum length for sentences to be considered")
    parser_extract.add_argument("-M", "--max-len-sentence", type=int, default=25,
                                help="maximum length for sentences to be considered")
    parser_extract.add_argument("-b", "--sentences-batch-size", type=int, default=100,
                                help="number of sentences in batch when extracting catenae")
    parser_extract.add_argument("-f", "--frequency-threshold", type=int, default=1,
                                help="frequency threshold applied in each batch for a catena to be kept")
    parser_extract.add_argument("--min-len-catena", type=int, default=0,
                                help="minimium length for each catena to be kept")
    parser_extract.add_argument("--max-len-catena", type=int, default=5,
                                help="maximum length for each catena to be kept. WARNING: highly impacts ram usage")
    parser_extract.set_defaults(func=_extract_catenae)

    # Weight list of catenae
    parser_weight = subparsers.add_parser('weight',
                                          description="compute weight function over catenae list",
                                          help="compute weight function over catenae list",
                                          formatter_class=ArgumentDefaultsHelpFormatter)
    parser_weight.add_argument("-o", "--output-dir", default="data/results/",
                               help="path to output dir")

    requiredWeight = parser_weight.add_argument_group('required arguments')
    requiredWeight.add_argument("-i", "--items-filepath", required=True)
    requiredWeight.add_argument("-t", "--totals-filepath", required=True)
    requiredWeight.add_argument("-c", "--catenae-filepath", required=True)

    parser_weight.set_defaults(func=_weight_catenae)

    # Extract list of accepted catenae
    parser_filter = subparsers.add_parser("filter",
                                          description="filter weighted list",
                                          help="filter weighted list",
                                          formatter_class=ArgumentDefaultsHelpFormatter)
    parser_filter.add_argument("-o", "--output-dir", default="data/results/",
                              help="path to output dir")
    parser_filter.add_argument("-i", "--input-filepath", required=True,
                               help="input filepath containing weighted list of catenae")
    parser_filter.add_argument('-f', "--min-freq", type=int, default=0,
                               help="minimum frequency for a catena to be kept")
    parser_filter.add_argument("-w", "--min-weight", type=float, default=float("-inf"),
                               help="minimum weight value for a catena to be kept")
    parser_filter.add_argument("-m", "--min-len-catena", type=int, default=0,
                               help="minimum catena length")
    parser_filter.add_argument("-M", "--max-len-catena", type=int, default=5,
                               help="maximum catena length")
    parser_filter.set_defaults(func=_filter_catenae)

    # Extract co-occurrences
    parser_coocc = subparsers.add_parser('cooccurrences',
                                         description="extract co-occurrences of catenae",
                                         help="extract co-occurrences of catenae",
                                         formatter_class=ArgumentDefaultsHelpFormatter)
    parser_coocc.add_argument("-o", "--output-dir", default="data/results/",
                              help="path to output dir, default is data/results/")
    parser_coocc.add_argument("-c", "--corpus-dirpath", default="data/corpus/",
                              help="path to corpus directory")
    parser_coocc.add_argument("-a", "--accepted-catenae",
                              help="filepth containing accepted catenae")
    parser_coocc.add_argument("-k", "--topk", type=float, default=float("inf"),
                              help="number of catenae to load")
    parser_coocc.add_argument("-m", "--min-len-sentence", type=int, default=1,
                              help="minimum length for sentences to be considered")
    parser_coocc.add_argument("-M", "--max-len-sentence", type=int, default=25,
                              help="maximum length for sentences to be considered")
    parser_coocc.add_argument("-b", "--sentences-batch-size", type=int, default=100,
                              help="number of sentences in batch when extracting catenae")
    parser_coocc.add_argument("-f", "--frequency-threshold", type=int, default=1,
                              help="frequency threshold applied in each batch for a catena to be kept")
    parser_coocc.add_argument("--min-len-catena", type=int, default=0,
                              help="minimium length for each catena to be kept")
    parser_coocc.add_argument("--max-len-catena", type=int, default=3,
                              help="maximum length for each catena to be kept. WARNING: highly impacts ram usage")
    parser_coocc.add_argument("--include-len-one-items", action='store_true',
                              help="include words regardless of their frequency and weight values")
    parser_coocc.add_argument("--words-filepath",
                              help="filepath to words list, to include if len one items are required")
    parser_coocc.set_defaults(func=_extract_cooccurrences)

    # Build DSM
    parser_dsm = subparsers.add_parser("build-dsm",
                                       description="build distributional space model",
                                       help="build distributional space model",
                                       formatter_class=ArgumentDefaultsHelpFormatter)
    parser_dsm.add_argument("-o", "--output-dir", default="data/results/",
                            help="path to output dir, default is data/results/")
    parser_dsm.add_argument("-c", "--cooccurrences-filepath", required=True)
    parser_dsm.add_argument("-f", "--frequences-filepath", required=True)
    parser_dsm.add_argument("-t", "--total", type=int, required=True)
    parser_dsm.set_defaults(func=_build_dsm)

    # Extract pairs of catenae at different abstraction levels
    parser_pairs = subparsers.add_parser("pairs",
                                         description="extract pairs of catenae",
                                         help="extract pairs of catenae",
                                         formatter_class=ArgumentDefaultsHelpFormatter)
    

    args = root_parser.parse_args()
    if "func" not in args:
        root_parser.print_usage()
        exit()
    args.func(args)