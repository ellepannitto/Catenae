"""_summary_
"""
import sys

from multiprocessing import Pool
from pathlib import Path

import catenae


output_dir_basename = sys.argv[1]
input_dir_basename = sys.argv[2]

FREQUENCY_THRESHOLD = 0
WEIGHT_THRESHOLD = 0
MIN_LEN_CATENA = 0
MAX_LEN_CATENA = 5


with Pool(3) as p:

    iter_output_dir = [Path(f"{output_dir_basename}/{x}/")
                       for x in [str(n).zfill(2) for n in range(1, 11)]] # TODO: change in all experiments
    iter_input_file = [f"{input_dir_basename}/{x}/catenae-weighted.gz"
                       for x in [str(n).zfill(2) for n in range(1, 11)]]

    iter_frequency_threshold = [FREQUENCY_THRESHOLD for _ in range(1, 11)]
    iter_weight_threshold = [WEIGHT_THRESHOLD for _ in range(1, 11)]
    iter_min_len_catena = [MIN_LEN_CATENA for _ in range(1, 11)]
    iter_max_len_catena = [MAX_LEN_CATENA for _ in range(1, 11)]

    p.starmap(catenae.core.extraction.filter_catenae,
              zip(iter_output_dir, iter_input_file,
                  iter_frequency_threshold, iter_weight_threshold,
                  iter_min_len_catena, iter_max_len_catena))
