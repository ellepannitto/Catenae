import sys
import catenae

from multiprocessing import Pool


output_dir_basename = sys.argv[1]
input_dir_basename = sys.argv[2]


with Pool(2) as p:

    output_dir_iter = [output_dir_basename+"{}/".format(x) for x in [str(n).zfill(2) for n in range(1, 11)]]
    input_dir_iter = [input_dir_basename+"{}/".format(x) for x in [str(n).zfill(2) for n in range(1, 11)]]

    cooccurrences_filepath_iter = [x+"/catenae-coocc-summed.gz" for x in input_dir_iter]
    frequences_filepath_iter = [x+"/catenae-freqs-summed.gz" for x in input_dir_iter]

    freqs_filepath_iter = [x+"/totals-freqs.txt" for x in input_dir_iter]

    TOT_iter = []

    for freqs_filepath in freqs_filepath_iter:
        with open(freqs_filepath) as fin:
            line = fin.readline().strip().split("\t")
            TOT = int(line[1])
            TOT_iter.append(TOT)

    p.starmap(catenae.core.dsm.build, zip(output_dir_iter, cooccurrences_filepath_iter, frequences_filepath_iter, TOT_iter))

# for i in range(1, 11):
#     n = str(i).zfill(2)

#     output_dir = output_dir_basename+"{}/".format(n)
#     input_dir = input_dir_basename+"{}/".format(n)

#     cooccurrences_filepath = input_dir+"/catenae-coocc-summed.gz"
#     frequences_filepath = input_dir+"/catenae-freqs-summed.gz"

#     freqs_filepath = input_dir+"/totals-freqs.txt"

#     with open(freqs_filepath) as fin:
#         line = fin.readline().strip().split("\t")
#         TOT = int(line[1])

#     catenae.core.dsm.build(output_dir, cooccurrences_filepath, frequences_filepath, TOT)