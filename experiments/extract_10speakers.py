import sys
import catenae

from multiprocessing import Pool

output_dir_basename = sys.argv[1]
input_dir_basename = sys.argv[2]

min_len_sentence = 3
max_len_sentence = 25
sentences_batch_size = 350
min_freq = 3
min_len_catena = 0
max_len_catena = 4

with Pool(3) as p:

    iter_input_dir = [input_dir_basename+"{}/".format(x) for x in [str(n).zfill(2) for n in range(1, 11)]]
    iter_output_dir = [output_dir_basename+"{}/".format(x) for x in [str(n).zfill(2) for n in range(1, 11)]]

    iter_min_len_sentence = [min_len_sentence for _ in range(1, 11)]
    iter_max_len_sentence = [max_len_sentence for _ in range(1, 11)]
    iter_sentences_batch_size = [sentences_batch_size for _ in range(1, 11)]
    iter_min_freq = [min_freq for _ in range(1, 11)]
    iter_min_len_catena = [min_len_catena for _ in range(1, 11)]
    iter_max_len_catena = [max_len_catena for _ in range(1, 11)]


    p.starmap(catenae.core.extraction.extract_catenae, zip(iter_output_dir, iter_input_dir, 
                                                           iter_min_len_sentence, iter_max_len_sentence, 
                                                           iter_sentences_batch_size, iter_min_freq,
                                                           iter_min_len_catena, iter_max_len_catena))