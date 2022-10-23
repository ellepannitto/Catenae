"""_summary_
"""
import sys

from multiprocessing import Pool
from pathlib import Path

import catenae


output_dir_basename = sys.argv[1]
corpus_dir_basename = sys.argv[2]
weighted_dir_basename = sys.argv[3]


with Pool(3) as p:

    output_dir_iter = [Path(f"{output_dir_basename}/{x}/")
                       for x in [str(n).zfill(2) for n in range(1, 11)]]
    corpus_dir_iter = [f"{corpus_dir_basename}/{x}/"
                       for x in [str(n).zfill(2) for n in range(1, 11)]]

    accepted_catenae_filepath_iter = [f"{weighted_dir_basename}/{x}/catenae-filtered.txt"
                                      for x in [str(n).zfill(2) for n in range(1, 11)]]

    top_k_iter = [float("inf") for _ in range(1, 11)]
    min_len_sentence_iter = [3 for _ in range(1, 11)]
    max_len_sentence_iter = [20 for _ in range(1, 11)]

    sentences_batch_size_iter = [5000 for _ in range(1, 11)]

    min_freq_iter = [2 for _ in range(1, 11)]

    min_len_catena_iter = [0 for _ in range(1, 11)]
    max_len_catena_iter = [3 for _ in range(1, 11)]

    include_words_iter = [True for _ in range(1, 11)]

    words_filepath_iter = [f"{weighted_dir_basename}/{x}/items-freq-summed.gz"
                           for x in [str(n).zfill(2) for n in range(1, 11)]]


    p.starmap(catenae.core.extraction.extract_coccurrences,
              zip(output_dir_iter, corpus_dir_iter, accepted_catenae_filepath_iter, top_k_iter,
                  min_len_sentence_iter, max_len_sentence_iter, sentences_batch_size_iter,
                  min_freq_iter, min_len_catena_iter, max_len_catena_iter,
                  include_words_iter, words_filepath_iter))

# for i in range(1, 11):
#     n = str(i).zfill(2)

#     output_dir = output_dir_basename+"{}/".format(n)
#     corpus_dir = corpus_dir_basename+"{}/".format(n)

#     accepted_catenae_filepath = weighted_dir_basename+"{}/catenae-filtered.txt".format(n)

#     top_k = float("inf")
#     min_len_sentence = 3
#     max_len_sentence = 20

#     sentences_batch_size = 5000

#     min_freq = 2

#     min_len_catena = 0
#     max_len_catena = 3

#     include_words = True

#     words_filepath = weighted_dir_basename+"{}/items-freq-summed.gz".format(n)


#     catenae.core.extraction.extract_coccurrences(output_dir, corpus_dir,
#                                     accepted_catenae_filepath, top_k,
#                                     min_len_sentence, max_len_sentence, sentences_batch_size,
#                                     min_freq, min_len_catena, max_len_catena,
#                                     include_words, words_filepath)
