import gzip
import math

from FileMerger.filesmerger import utils as fmergerutils

def build(output_dir, coocc_filepath, freqs_filepath, TOT):

    with fmergerutils.open_file_by_extension(coocc_filepath) as fin_cocc, \
        fmergerutils.open_file_by_extension(freqs_filepath) as fin_freqs_left, \
        fmergerutils.open_file_by_extension(freqs_filepath) as fin_freqs_right, \
        gzip.open(output_dir + "catenae-ppmi.gz", "wt") as fout:

        lineno = 1

        line_cocc = fin_cocc.readline()

        line_freq_left = fin_freqs_left.readline()
        # new_start_from = fin_freqs_left.tell()
        cat_l, freq_l = line_freq_left.strip().split("\t")
        freq_l = float(freq_l)

        line_freq_right = fin_freqs_right.readline()
        cat_r, freq_r = line_freq_right.strip().split("\t")
        freq_r = float(freq_r)


        while line_cocc:

            cats, freq = line_cocc.strip().split("\t")
            cat1, cat2 = cats.split(" ")
            freq = float(freq)

            # print(cat1, cat2, freq, "---", cat_l, cat_l<cat1, "---", cat_r, cat_r<cat2)

            while cat_l < cat1:
                line_freq_left = fin_freqs_left.readline()
                # first_start_from = new_start_from
                # new_start_from = fin_freqs_left.tell()
                cat_l, freq_l, = line_freq_left.strip().split("\t")
                # print("HERE new cat l:", cat_l)
                freq_l = float(freq_l)

            if cat_r > cat2:
                fin_freqs_right = fmergerutils.open_file_by_extension(freqs_filepath)
                line_freq_right = fin_freqs_right.readline()
                cat_r, freq_r = line_freq_right.strip().split("\t")
                freq_r = float(freq_r)


            while cat_r < cat2:
                line_freq_right = fin_freqs_right.readline()
                cat_r, freq_r = line_freq_right.strip().split("\t")
                freq_r = float(freq_r)

            assert cat1 == cat_l, "MISSING CATENA"
            assert cat2 == cat_r, "MISSING CATENA"

            ppmi = freq * math.log(freq*TOT/(freq_l*freq_r))
            if ppmi > 0:
                print("{}\t{}\t{}\t{}".format(cat1, cat2, freq, ppmi), file=fout)
                # print(cat1, cat2, freq, "---", cat_l, freq_l, "---", cat_r, freq_r, "---", ppmi)
            # else:
                # print("REMOVE", cat1, cat2, freq)
            # input()

            line_cocc = fin_cocc.readline()
            lineno += 1

            if not lineno%10000:
                print("PROCESSING LINE", lineno)