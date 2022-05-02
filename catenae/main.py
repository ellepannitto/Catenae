import argparse
from argparse import RawTextHelpFormatter, ArgumentDefaultsHelpFormatter
import logging.config
import os

from catenae.utils import config_utils as cutils
from catenae.core import extraction as extraction
from catenae.core import statistics as stats
from catenae.core import dsm as dsm
from catenae.core import corpus as corpus

config_dict = cutils.load(os.path.join(os.path.dirname(__file__), 'logging_utils', 'logging.yml'))
logging.config.dictConfig(config_dict)

logger = logging.getLogger(__name__)


def _compute_pos_stats(args):
    output_dir = args.output_dir
    corpus_dir = args.corpus_dirpath
    pos_list = args.pos

    stats.compute_pos_distribution(output_dir, corpus_dir, pos_list)


def _compute_morph_stats(args):
    output_dir = args.output_dir
    corpus_dir = args.corpus_dirpath
    trait = args.trait
    values_list = args.values

    stats.compute_morph_distribution(output_dir, corpus_dir, trait, values_list)


# def _compute_mood_stats(args):
#     output_dir = args.output_dir
#     corpus_dir = args.corpus_dirpath
#     mood_list = args.mood
#
#     stats.compute_mood_distribution(output_dir, corpus_dir, mood_list)
#
#
# def _compute_form_stats(args):
#     output_dir = args.output_dir
#     corpus_dir = args.corpus_dirpath
#     form_list = args.form
#
#     stats.compute_tense_distribution(output_dir, corpus_dir, form_list)


def _compute_verbedges_stats(args):
    output_dir = args.output_dir
    corpus_dir = args.corpus_dirpath
    edges = args.number_edges

    stats.compute_verbedges_distribution(output_dir, corpus_dir, edges)


def _compute_sbj_stats(args):
    output_dir = args.output_dir
    corpus_dir = args.corpus_dirpath

    stats.compute_sbj_distribution(output_dir, corpus_dir)


def _compute_synrel_stats(args):
    output_dir = args.output_dir
    corpus_dir = args.corpus_dirpath
    synrel_list = args.synrels

    stats.compute_synrel_distribution(output_dir, corpus_dir, synrel_list)


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


def _sample_input(args):
    output_dir = args.output_dir
    input_dir = args.corpus_dirpath
    size = args.size
    seed = args.seed

    corpus.sample(output_dir, input_dir, size, seed)


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

    parser_morph_stats = subparsers.add_parser('morph-stats',
                                               description='compute list of stats for a corpus',
                                               help='compute list of stats for a corpus',
                                               formatter_class=ArgumentDefaultsHelpFormatter)
    parser_morph_stats.add_argument("-o", "--output-dir", default="data/stats/",
                                    help="path to output dir")
    parser_morph_stats.add_argument("-c", "--corpus-dirpath", default="data/corpus/",
                                    help="path to corpus directory")
    parser_morph_stats.add_argument("-t", "--trait", required=True,
                                    help="Morphological trait name")
    parser_morph_stats.add_argument("-v", "--values", required=True, nargs="+",
                                    help="Morphological trait values")
    parser_morph_stats.set_defaults(func=_compute_morph_stats)

    parser_verbedges_stats = subparsers.add_parser('verbedges-stats',
                                                   description='compute list of stats for a corpus',
                                                   help='compute list of stats for a corpus',
                                                   formatter_class=ArgumentDefaultsHelpFormatter)
    parser_verbedges_stats.add_argument("-o", "--output-dir", default="data/stats/",
                                        help="path to output dir")
    parser_verbedges_stats.add_argument("-c", "--corpus-dirpath", default="data/corpus/",
                                        help="path to corpus directory")
    parser_verbedges_stats.add_argument("-n", "--number-edges", required=True, type=int,
                                        help="Number of edges for verb")
    parser_verbedges_stats.set_defaults(func=_compute_verbedges_stats)

    parser_sbj_stats = subparsers.add_parser('subj-stats',
                                             description='compute list of stats for a corpus',
                                             help='compute list of stats for a corpus',
                                             formatter_class=ArgumentDefaultsHelpFormatter)
    parser_sbj_stats.add_argument("-o", "--output-dir", default="data/stats/",
                                  help="path to output dir")
    parser_sbj_stats.add_argument("-c", "--corpus-dirpath", default="data/corpus/",
                                  help="path to corpus directory")
    parser_sbj_stats.set_defaults(func=_compute_sbj_stats)

    parser_synrel_stats = subparsers.add_parser('synrel-stats',
                                                description='compute list of stats for a corpus',
                                                help='compute list of stats for a corpus',
                                                formatter_class=ArgumentDefaultsHelpFormatter)
    parser_synrel_stats.add_argument("-o", "--output-dir", default="data/stats/",
                                     help="path to output dir")
    parser_synrel_stats.add_argument("-c", "--corpus-dirpath", default="data/corpus/",
                                     help="path to corpus directory")
    parser_synrel_stats.add_argument("-r", "--synrels", required=True, nargs="+",
                                     help="Universal Syntactic Relation tag")
    parser_synrel_stats.set_defaults(func=_compute_synrel_stats)

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

    # Sample input of determinate size based on number of words
    parser_sample = subparsers.add_parser("sample-input",
                                          description="sample input of set size",
                                          help="sample input of set size",
                                          formatter_class=ArgumentDefaultsHelpFormatter)
    parser_sample.add_argument("-o", "--output-dir", default="data/input_sampled/",
                               help="path to output dir, default is data/input_sampled/")
    parser_sample.add_argument("-c", "--corpus-dirpath", default="data/corpus/",
                               help="path to corpus directory")
    parser_sample.add_argument("-s", "--size", type=int, required=True,
                               help="number of words to include in the sample")
    parser_sample.add_argument("--seed", type=int, default=42)
    parser_sample.set_defaults(func=_sample_input)

    # # Extract pairs of catenae at different abstraction levels
    # parser_pairs = subparsers.add_parser("pairs",
    #                                      description="extract pairs of catenae",
    #                                      help="extract pairs of catenae",
    #                                      formatter_class=ArgumentDefaultsHelpFormatter)
    #
    args = root_parser.parse_args()
    if "func" not in args:
        root_parser.print_usage()
        exit()
    args.func(args)
