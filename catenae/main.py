import argparse
import logging.config
import os

from catenae.utils import config_utils as cutils
from catenae.core import extraction as extraction

config_dict = cutils.load(os.path.join(os.path.dirname(__file__), 'logging_utils', 'logging.yml'))
logging.config.dictConfig(config_dict)

logger = logging.getLogger(__name__)


def _extract_catenae(args):
    output_dir = args.output_dir
    corpus_dir = args.corpus_dirpath

    min_len_sentence = 1
    max_len_sentence = 25
    sentences_batch_size = 5
    min_freq = 3


    extraction.extract_catenae(output_dir, corpus_dir,
                               min_len_sentence, max_len_sentence, sentences_batch_size,
                               min_freq)


def main():
    root_parser = argparse.ArgumentParser(prog='catenae')
    subparsers = root_parser.add_subparsers(title="actions", dest="actions")

#    if len(sys.argv) == 1:
#        root_parser.print_help(sys.stderr)
#        sys.exit(1)

    # FIRST
    parser_extract = subparsers.add_parser('extract',
                                           description='extract list of catenae',
                                           help='extract list of catenae')
    parser_extract.add_argument("-o", "--output-dir", default="data/results/",
                                help="path to output dir, default is data/results/")
    parser_extract.add_argument("-c", "--corpus-dirpath", default="data/corpus/",
                                help="path to corpus directory")
    parser_extract.set_defaults(func=_extract_catenae)

    args = root_parser.parse_args()
    if "func" not in args:
        root_parser.print_usage()
        exit()
    args.func(args)