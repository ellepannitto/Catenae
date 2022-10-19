import os
import sys
import logging
import glob
import random

import ufal.udpipe as udpipe

from catenae.utils import corpus_utils as cutils

logger = logging.getLogger(__name__)


def reservoir_tokens_number(files, size, seed=42):

    random.seed(seed)

    res = []
    len_res = []

    considered_tokens = 0
    sentence_number = -1
    for filename in files:
        logger.info("reading file {}".format(filename))

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

    x = 0
    i = 0
    while x < size and i < len(res):
        x += len_res[i]
        i = i+1

    return list(sorted(res[:i]))


def sample(output_dir, input_dir, size, seed):

    input_files = glob.glob(input_dir+"/*")
    random.Random(seed).shuffle(input_files)

    L = len(input_files)

    training_files, valid_files, test_files = input_files[:int(L*0.8)], \
                                              input_files[int(L*0.8):int(L*0.9)], \
                                              input_files[int(L*0.9):]

    training_size = int(size)
    valid_size, test_size = int(size*0.1), int(size*0.1)

    logger.info("Setting training size to {}, valid size to {} and test size to \
                {} words".format(training_size, valid_size, test_size))

    sentences_for_training = reservoir_tokens_number(training_files, training_size)
    sentences_for_valid = reservoir_tokens_number(valid_files, valid_size)
    sentences_for_test = reservoir_tokens_number(test_files, test_size)

    logger.info("end reservoir")

    print_to_file(sentences_for_training, training_files, output_dir, "train")
    print_to_file(sentences_for_valid, valid_files, output_dir, "valid")
    print_to_file(sentences_for_test, test_files, output_dir, "test")


def print_to_file(sentences_idxs, files, output_path, str):
    index = 0
    sentence_number = -1

    logger.info("LEN SENTENCES RESERVOIR: {}".format(len(sentences_idxs)))

    with open(output_path+str+".txt", "w") as fout_linear, open(output_path+str+".conll", "w") as fout_conll:
        for filename in files:

            logger.info("reading file {}...".format(filename))

            for sentence in cutils.plain_conll_reader(filename):
                sentence_number += 1

                if index < len(sentences_idxs) and sentence_number == sentences_idxs[index]:

                    print("\n".join(sentence)+"\n", file=fout_conll)
                    print(cutils.get_linear(sentence)+"\n", file=fout_linear)

                    index += 1
                    if not sentence_number % 100:
                        logger.info("sentence n. {}...".format(sentence_number))

                    if index == len(sentences_idxs):
                        break


def parse(output_dir, input_dir, model_path):
    model = udpipe.Model.load(model_path)

    pipeline = udpipe.Pipeline(model, "horizontal", udpipe.Pipeline.DEFAULT, udpipe.Pipeline.DEFAULT, "conllu")
    error = udpipe.ProcessingError()

    # Read whole input
    for filename in os.listdir(input_dir):
        basename = filename.split(".")[0]

        logger.info("processing file %s/%s", input_dir, filename)

        with open(input_dir+filename) as fin, \
            open(output_dir.joinpath(f"{basename}.conllu"), "w") as fout:
            text = ''.join(fin.readlines())

            # Process data
            processed = pipeline.process(text, error)
            if error.occurred():
                logger.info("An error occurred when running run_udpipe: ")
                logger.info(error.message)
                sys.exit(1)

            print(processed, file=fout)