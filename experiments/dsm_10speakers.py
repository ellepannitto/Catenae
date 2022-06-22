import sys
import catenae


output_dir_basename = sys.argv[1]
input_dir_basename = sys.argv[2]

for i in range(1, 11):
    n = str(i).zfill(2)

    output_dir = output_dir_basename+"{}/".format(n)
    input_dir = input_dir_basename+"{}/".format(n)

    cooccurrences_filepath = input_dir+"/catenae-coocc-summed.gz"
    frequences_filepath = input_dir+"/catenae-freqs-summed.gz"

    freqs_filepath = input_dir+"/totals-freqs.txt"

    with open(freqs_filepath) as fin:
        line = fin.readline().strip().split("\t")
        TOT = int(line[1])

    catenae.core.dsm.build(output_dir, cooccurrences_filepath, frequences_filepath, TOT)