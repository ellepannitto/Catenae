# pylint: disable=unspecified-encoding
"""_summary_

Returns:
    _type_: _description_
"""
import os
import sys
import logging
import glob
import random

import ufal.udpipe as udpipe # pylint: disable=consider-using-from-import

from catenae.utils import corpus_utils as cutils


logger = logging.getLogger(__name__)


def reservoir_tokens_number(files, size, seed=42):

    random.seed(seed)

    res = []
    len_res = []

    considered_tokens = 0
    sentence_number = -1
    for filename in files:
        logger.info("reading file %s", filename)

        for sentence in cutils.plain_conll_reader(filename):

            sentence_number += 1
            considered_tokens += len(sentence)

            if considered_tokens < size*1.2:
                res.append(sentence_number)
                len_res.append(len(sentence))
            else:
                j = random.randrange(sentence_number)
                if j < len(res):
                    res[j] = sentence_number
                    len_res[j] = len(sentence)

    cur_size = 0
    cur_len = 0
    while cur_size < size and cur_len < len(res):
        cur_size += len_res[cur_len]
        cur_len = cur_len+1

    return list(sorted(res[:cur_len]))


def sample(output_dir, input_dir, size, seed):

    input_files = glob.glob(input_dir+"/*")
    random.Random(seed).shuffle(input_files)

    L = len(input_files) # pylint:disable=invalid-name

    training_files, valid_files, test_files = input_files[:int(L*0.8)], \
                                              input_files[int(L*0.8):int(L*0.9)], \
                                              input_files[int(L*0.9):]

    training_size = int(size)
    valid_size, test_size = int(size*0.1), int(size*0.1)

    logger.info("Setting training size to %d, valid size to %d and test size to \
                %d words", training_size, valid_size, test_size)

    sentences_for_training = reservoir_tokens_number(training_files, training_size)
    sentences_for_valid = reservoir_tokens_number(valid_files, valid_size)
    sentences_for_test = reservoir_tokens_number(test_files, test_size)

    logger.info("end reservoir")

    print_to_file(sentences_for_training, training_files, output_dir, "train")
    print_to_file(sentences_for_valid, valid_files, output_dir, "valid")
    print_to_file(sentences_for_test, test_files, output_dir, "test")


def print_to_file(sentences_idxs, files, output_path, basename):
    index = 0
    sentence_number = -1

    logger.info("LEN SENTENCES RESERVOIR: %d", len(sentences_idxs))

    with open(output_path / basename / ".txt", "w") as fout_linear, \
        open(output_path / basename / ".conll", "w") as fout_conll:

        for filename in files:
            logger.info("reading file %s...", filename)

            for sentence in cutils.plain_conll_reader(filename):
                sentence_number += 1

                if index < len(sentences_idxs) and sentence_number == sentences_idxs[index]:

                    print("\n".join(sentence)+"\n", file=fout_conll)
                    print(cutils.get_linear(sentence)+"\n", file=fout_linear)

                    index += 1
                    if not sentence_number % 100:
                        logger.info("sentence n. %d...", sentence_number)

                    if index == len(sentences_idxs):
                        break


def parse(output_dir, input_dir, model_path):

    model = udpipe.Model.load(model_path)

    pipeline = udpipe.Pipeline(model, "horizontal",
                               udpipe.Pipeline.DEFAULT, udpipe.Pipeline.DEFAULT, "conllu")
    error = udpipe.ProcessingError()

    # Read whole input
    for filename in os.listdir(input_dir):
        basename = filename.split(".")[0]

        logger.info("processing file %s/%s", input_dir, filename)

        with open(input_dir / filename) as fin, \
            open(output_dir.joinpath(f"{basename}.conllu"), "w") as fout:
            text = ''.join(fin.readlines())

            # Process data
            processed = pipeline.process(text, error)
            if error.occurred():
                logger.info("An error occurred when running run_udpipe: ")
                logger.info(error.message)
                sys.exit(1)

            print(processed, file=fout)
