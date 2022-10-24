import sys

from multiprocessing import Pool

import catenae

output_dir_basename = sys.argv[1]
input_dir_basename = sys.argv[2]


with Pool(3) as p:

    iter_output_dir = [output_dir_basename+"{}/".format(x) for x in [str(n).zfill(2) for n in range(1, 11)]]

    iter_items_fpaths = [input_dir_basename+"{}/items-freq-summed.gz".format(x) for x in [str(n).zfill(2) for n in range(1, 11)]]
    iter_totals_fpaths = [input_dir_basename+"{}/totals-freq-summed.gz".format(x) for x in [str(n).zfill(2) for n in range(1, 11)]]
    iter_catenae_fpaths = [input_dir_basename+"{}/catenae-freq-summed.gz".format(x) for x in [str(n).zfill(2) for n in range(1, 11)]]

    p.starmap(catenae.core.extraction.weight_catenae, zip(iter_output_dir,
                                                          iter_items_fpaths, iter_totals_fpaths, iter_catenae_fpaths))