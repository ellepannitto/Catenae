import sys
import catenae

from multiprocessing import Pool

input_dir_basename = sys.argv[1]
output_dir_basename = sys.argv[2]
model_path = sys.argv[1]




with Pool(5) as p:

    iter_input_dir = [input_dir_basename+"{}/".format(str(n).zfill(2) for n in range(1, 11))]
    iter_output_dir = [output_dir_basename+"{}/".format(str(n).zfill(2) for n in range(1, 11))]
    iter_model = [model_path for _ in range(1, 11)]

    for el in zip(iter_output_dir, iter_input_dir, iter_model):
        print(el)

#    print(p.starmap(catenae.core.corpus.parse, zip(iter_output_dir, iter_input_dir, iter_model)))