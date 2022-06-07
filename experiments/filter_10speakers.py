import sys
import catenae

from multiprocessing import Pool

output_dir_basename = sys.argv[1]
input_dir_basename = sys.argv[2]

frequency_threshold = 200
weight_threshold = 20
min_len_catena = 0
max_len_catena = 5

with Pool(3) as p:
    
    iter_output_dir = [output_dir_basename+"{}/".format(x) for x in [str(n).zfill(2) for n in range(1, 11)]]
    iter_input_file = [input_dir_basename+"{}/catenae-weighted.gz".format(x) for x in [str(n).zfill(2) for n in range(1, 11)]]

    iter_frequency_threshold = [frequency_threshold for _ in range(1, 11)]
    iter_weight_threshold = [weight_threshold for _ in range(1, 11)]
    iter_min_len_catena = [min_len_catena for _ in range(1, 11)]
    iter_max_len_catena = [max_len_catena for _ in range(1, 11)]


    p.starmap(catenae.core.extraction.filter_catenae, zip(iter_output_dir, iter_input_file, 
                                                          iter_frequency_threshold, iter_weight_threshold, 
                                                          iter_min_len_catena, iter_max_len_catena))