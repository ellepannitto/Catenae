# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
"""
Entry point for package "catenae"
"""

# TODO add function to check input dirs and input filepaths

import argparse
from argparse import RawTextHelpFormatter, ArgumentDefaultsHelpFormatter
import logging.config
import os
import sys

from catenae.utils import config_utils as cutils
from catenae.utils import main_utils as mutils

config_dict = cutils.load(os.path.join(os.path.dirname(__file__), 'logging_utils', 'logging.yml'))
logging.config.dictConfig(config_dict)

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for package "catenae"

    The function deals with parsing arguments and dispatching them to the proper functions.

    """

    root_parser = argparse.ArgumentParser(prog='catenae', formatter_class=RawTextHelpFormatter)
    subparsers = root_parser.add_subparsers(title="actions", dest="actions")

    # Compute statistics on corpus
    # Part-of-Speech statistics
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
    parser_pos_stats.set_defaults(func=mutils.compute_pos_stats)

    # Morphology statistics
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
    parser_morph_stats.set_defaults(func=mutils.compute_morph_stats)

    # Statistics on distribution of verb edges
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
    parser_verbedges_stats.set_defaults(func=mutils.compute_verbedges_stats)

    # Statistics on distribution of subjects
    parser_sbj_stats = subparsers.add_parser('subj-stats',
                                             description='compute list of stats for a corpus',
                                             help='compute list of stats for a corpus',
                                             formatter_class=ArgumentDefaultsHelpFormatter)
    parser_sbj_stats.add_argument("-o", "--output-dir", default="data/stats/",
                                  help="path to output dir")
    parser_sbj_stats.add_argument("-c", "--corpus-dirpath", default="data/corpus/",
                                  help="path to corpus directory")
    parser_sbj_stats.set_defaults(func=mutils.compute_sbj_stats)

    # Statistics on distribution of syntactic relations
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
    parser_synrel_stats.set_defaults(func=mutils.compute_synrel_stats)

    # Extraction of catenae
    parser_extract = subparsers.add_parser('extract-catenae',
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
                                help="frequency threshold applied in each batch for \
                                a catena to be kept")
    parser_extract.add_argument("--min-len-catena", type=int, default=0,
                                help="minimium length for each catena to be kept")
    parser_extract.add_argument("--max-len-catena", type=int, default=5,
                                help="maximum length for each catena to be kept. \
                                    WARNING: highly impacts ram usage")
    parser_extract.set_defaults(func=mutils.extract_catenae)

    # Weight list of catenae
    parser_weight = subparsers.add_parser('weight-catenae',
                                          description="compute weight function over catenae list",
                                          help="compute weight function over catenae list",
                                          formatter_class=ArgumentDefaultsHelpFormatter)
    parser_weight.add_argument("-o", "--output-dir", default="data/results/",
                               help="path to output dir")

    required_weight = parser_weight.add_argument_group('required arguments')
    required_weight.add_argument("-i", "--items-filepath", required=True)
    required_weight.add_argument("-t", "--totals-filepath", required=True)
    required_weight.add_argument("-c", "--catenae-filepath", required=True)

    parser_weight.set_defaults(func=mutils.weight_catenae)

    # Extract list of accepted catenae
    parser_filter = subparsers.add_parser("filter-catenae",
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
    parser_filter.set_defaults(func=mutils.filter_catenae)

    # Extract co-occurrences
    parser_coocc = subparsers.add_parser('extract-cooccurrences',
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
                              help="frequency threshold applied in each batch \
                              for a catena to be kept")
    parser_coocc.add_argument("--min-len-catena", type=int, default=0,
                              help="minimium length for each catena to be kept")
    parser_coocc.add_argument("--max-len-catena", type=int, default=3,
                              help="maximum length for each catena to be kept.\
                              WARNING: highly impacts ram usage")
    parser_coocc.add_argument("--include-len-one-items", action='store_true',
                              help="include words regardless of their frequency and weight values")
    parser_coocc.add_argument("--words-filepath",
                              help="filepath to words list, to include if items \
                              of length one are required")
    parser_coocc.set_defaults(func=mutils.extract_cooccurrences)

    # Build DSM
    parser_dsm = subparsers.add_parser("build-dsm",
                                       description="build distributional space model",
                                       help="build distributional space model",
                                       formatter_class=ArgumentDefaultsHelpFormatter)
    parser_dsm.add_argument("-o", "--output-dir", default="data/results/",
                            help="path to output dir, default is data/results/")
    parser_dsm.add_argument("-c", "--cooccurrences-filepath",
                            help="path to cooccurrences file", required=True)
    parser_dsm.add_argument("-f", "--frequences-filepath",
                            help="path to frequencies file", required=True)
    parser_dsm.add_argument("-t", "--total", type=int, required=True,
                            help="sum of frequencies")
    parser_dsm.set_defaults(func=mutils.build_dsm)

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
    parser_sample.set_defaults(func=mutils.sample_input)

    # Parse input through UDPipe (using Universal Dependencies labels)
    parser_udparse = subparsers.add_parser("udparse",
                                           description="parse data with UD format",
                                           help="parse data with UD format",
                                           formatter_class=ArgumentDefaultsHelpFormatter)
    parser_udparse.add_argument("-o", "--output-dir", default="data/output_parsed/",
                               help="path to output dir, default is data/output_parsed/")
    parser_udparse.add_argument("-i", "--input-dir", required=True,
                               help="path to input dir")
    parser_udparse.add_argument("-m", "--model", required=True,
                                help="path to file containing model for udpipe lib")
    parser_udparse.set_defaults(func=mutils.udparse)

    # Compute spearmanR correlation among lists of catenae
    parser_correlate = subparsers.add_parser("spearman",
                                            description="compute spearman correlation between lists of catenae", # pylint:disable=line-too-long
                                            help="compute spearman correlation between lists of catenae", # pylint:disable=line-too-long
                                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser_correlate.add_argument("-o", "--output-dir", default="data/output_correlations/",
                                  help="path to output dir, default is data/output_correlations/")
    parser_correlate.add_argument("-k", "--top-k", type=int, default=10_000,
                                  help="number of structures to correlate")
    parser_correlate.add_argument("--mi-threshold", type=int, default=0,
                                  help="theshold on mutual information")
    parser_correlate.add_argument("--frequency-threshold", type=int, default=20,
                                  help="threshold on frequency")
    parser_correlate.add_argument("-l", "--files-list", required=True, nargs="+",
                                  help="list of filenames to be used as input")
    parser_correlate.set_defaults(func=mutils.correlate)

    # Extract core catenae
    # TODO: check!
    parser_corecatenae = subparsers.add_parser("corecatenae",
                                               description="compute stats about the core catenae for group of speakers", # pylint:disable=line-too-long
                                               help="compute stats about the core catenae for group of speakers", # pylint:disable=line-too-long
                                               formatter_class=ArgumentDefaultsHelpFormatter)
    parser_corecatenae.add_argument("-o", "--output-dir", default="data/output_corecatenae/",
                                    help="path to output dir, default is data/output_corecatenae/")
    parser_corecatenae.add_argument("-i", "--input-files-list", required=True, nargs="+",
                                    help="list of filenames to be used as input")
    parser_corecatenae.add_argument("-b", "--babbling-files-list", required=True, nargs="+",
                                    help="list of filenames produced in the babbling phase to be \
                                    used as input")
    parser_corecatenae.add_argument("-k", "--top-k", type=int, default=10_000,
                                    help="threshold")
    parser_corecatenae.set_defaults(func=mutils.corecatenae)

    # Extract sentences containing specific catenae
    parser_extractsents = subparsers.add_parser("extract-sentences",
                                                description="extract sentences containing catena",
                                                help="extract sentences containing catena",
                                                formatter_class=ArgumentDefaultsHelpFormatter)
    parser_extractsents.add_argument("-o", "--output-dir", default="data/output_sentences/",
                                     help="path to output dir, default is data/output_sentences/")
    parser_extractsents.add_argument("-i", "--input-dir", required=True,
                                     help="directory containing parsed files to be used as input")
    parser_extractsents.add_argument("-c", "--catenae-list", nargs="+",
                                     help="list of catenae to look for")
    parser_extractsents.set_defaults(func=mutils.extract_sentences)

    # Compute similarity matrix for given DSM
    parser_simmatrix = subparsers.add_parser("similarity-matrix",
                                             description="compute full similarity matrix for dsm",
                                             help="compute full similarity matrix for dsm",
                                             formatter_class=ArgumentDefaultsHelpFormatter)
    parser_simmatrix.add_argument("-o", "--output-dir", default="data/output_simmatrix/",
                                  help="path to output dir, default is data/output_simmatrix/")
    parser_simmatrix.add_argument("-s", "--dsm-vec", required=True,
                                  help="path to file containing distributional space vectors")
    parser_simmatrix.add_argument("-i", "--dsm-idx", required=True,
                                  help="path to file containing distributional space indexes")
    parser_simmatrix.add_argument("--reduced-left-matrix", default="all",
                                  help="optional path to first subset of items")
    parser_simmatrix.add_argument("--reduced-right-matrix", default="all",
                                  help="optional path to second subset of items")
    parser_simmatrix.add_argument("--chunked", action='store_true',
                                  help="set to True for chunked version, ram-memory efficient")
    parser_simmatrix.add_argument("--working-memory", default=16_000, type=int,
                                  help="Working memory (MiB) for pairwise computation")
    parser_simmatrix.set_defaults(func=mutils.compute_simmatrix)

    # Reduce similarity matrix to top-k neighbors
    parser_reducematrix = subparsers.add_parser("reduce-simmatrix",
                                                description="reduce simmatrix to top K neighbors",
                                                help="reduce simmatrix to top K neighbors",
                                                formatter_class=ArgumentDefaultsHelpFormatter)
    parser_reducematrix.add_argument("-o", "--output-dir",
                                     default="data/output_reducedsimmatrix/",
                                     help="path to output dir, \
                                     default is data/output_reducedsimmatrix/")
    parser_reducematrix.add_argument("--similarities-values", required=True, nargs="+",
                                     help="paths to gzipped file containing cosine similarities")
    parser_reducematrix.add_argument("-k", "--top-k", default=10_000, type=int,
                                     help="number of dimensions to keep")
    parser_reducematrix.set_defaults(func=mutils.reduce_simmatrix)

    # Query nearest neighbors given a similarity matrix
    parser_queryneighbors = subparsers.add_parser("query-neighbors",
                                                  description="query nearest neighbors for catena",
                                                  help="query nearest neighbors for catena",
                                                  formatter_class=ArgumentDefaultsHelpFormatter)
    parser_queryneighbors.add_argument("-s", "--similarities-matrix", required=True,
                                       help="path to file containing similarity values")
    parser_queryneighbors.add_argument("-i", "--indexes-matrix", required=True,
                                       help="path to file containing indexes values")
    parser_queryneighbors.add_argument("-m", "--indexes-map", required=True,
                                       help="path to file containing catena to index mapping")
    parser_queryneighbors.set_defaults(func=mutils.query_neighbors)

    # Creates matrix to project list of catenae on data
    parser_glassmatrix = subparsers.add_parser("glassify-matrix",
                                               description="filters parsed file based on set of catenae", # pylint:disable=line-too-long
                                               help="filters parsed file based on set of catenae",
                                               formatter_class=ArgumentDefaultsHelpFormatter)
    parser_glassmatrix.add_argument("-o", "--output-dir", default="data/output_glass/",
                                    help="path to output dir, default is data/output_glass/")
    parser_glassmatrix.add_argument("-i", "--input", required=True,
                                    help="path to input file in CoNLL format")
    parser_glassmatrix.add_argument("-c", "--catenae", required=True,
                                    help="path to file containing catenae")
    parser_glassmatrix.set_defaults(func=mutils.glassify_matrix)


    # Collapses matrix to project list of catenae on data
    parser_glasscollapse = subparsers.add_parser("glassify-collapse",
                                                 description="collapses glassified matrices",
                                                 help="collapses glassified matrices",
                                                 formatter_class=ArgumentDefaultsHelpFormatter)
    parser_glasscollapse.add_argument("-o", "--output-dir", default="data/output_glasscollapse/",
                                      help="path to output dir, default is data/output_glasscollapse/") # pylint:disable=line-too-long
    parser_glasscollapse.add_argument("-i", "--input-dir", required=True,
                                      help="path to input folder containing matrices")
    parser_glasscollapse.add_argument("--multiprocess", action="store_true")
    parser_glasscollapse.add_argument("--n-workers", type=int, default=4,
                                      help="number of processes")
    parser_glasscollapse.add_argument("--chunksize", type=int, default=10,
                                      help="size of chunks for multiprocess")
    parser_glasscollapse.set_defaults(func=mutils.glassify_collapse)


    args = root_parser.parse_args()


    # printing summary of parameters
    options = vars(args)
    summary_str = f"CALLING {options['actions']} WITH PARAMETERS:\n"
    for arg_name, arg_value in options.items():
        if not arg_name in ["func", "actions"]:
            summary_str+=f"\t{arg_name}\t\t--- VALUE: {arg_value}\n"
    print(summary_str)


    if "func" not in args:
        root_parser.print_usage()
        sys.exit()
    args.func(args)
