import logging
import collections

from scipy.stats.mstats import spearmanr

logger = logging.getLogger(__name__)


def correlate(output_dir, filenames_list, topk):
    
    print(filenames_list)
    input()

    catdict = {}

    catdict_lists = {}

    for filename in filenames_list:
        with open(filename) as fin:
            catdict[filename] = {}

            first_mi = 100
            while first_mi > 20:
                line = fin.readline()
                linesplit = line.strip().split("\t")
                catena = linesplit[0].lower()
                freq = float(linesplit[1])
                mi = float(linesplit[2])
                if freq > 200:
                    catdict[filename][catena] = mi
                    first_mi = mi

    for filename in catdict:
        catdict_lists[filename] = list(sorted(catdict[filename].items(), key=lambda x: -x[1]))


    # SPEARMAN
    stats = {}
    p_values = {}
    keys_list = set()
    zeros = {}
    vectors = {}

    for filename in catdict:

        stats[filename] = collections.defaultdict(lambda: -1.0)
        p_values[filename] = collections.defaultdict(lambda: -1.0)
        keys_list.add(filename)

        dims = catdict_lists[filename][:topk]

        for filename2 in catdict:
            vectors[filename] = []
            vectors[filename2] = []
            zeros[filename2] = 0

            for catena, mi in dims:

                vectors[filename].append(mi)

                if catena in catdict[filename2]:
                    vectors[filename2].append(catdict[filename2][catena])
                else:
                    vectors[filename2].append(0)
                    zeros[filename2] += 1

            s, p_s = spearmanr(vectors[filename], vectors[filename2])
            stats[filename][filename2] = s
            p_values[filename][filename2] = p_s


    with open(output_dir+"/spearmanr.txt", "w") as fout:
        for filename in stats:
            for filename2 in stats[filename]:
                s, p_s = stats[filename][filename2], p_values[filename][filename2]
                print("{}\t{}\t{}\t{}".format(filename, filename2, s, p_s), file=fout)
