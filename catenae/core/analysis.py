import logging
import collections
import gzip

from scipy.stats.mstats import spearmanr

logger = logging.getLogger(__name__)


def correlate(output_dir, filenames_list, topk, mi_threshold, fr_threshold):
    
    #print("HELLO")
    #print(filenames_list)
    #input()

    catdict = {}
    catdict_lists = {}

    for filename in filenames_list:
        with gzip.open(filename, "rt") as fin:
            catdict[filename] = {}

            first_mi = 100
            line = fin.readline()
            
            for line in fin:
            #while first_mi > mi_threshold:
                #print(line)
#                line = fin.readline()
                linesplit = line.strip().split("\t")
                catena = linesplit[0].lower()
                freq = float(linesplit[1])
                mi = float(linesplit[2])
                if freq > fr_threshold:
                    catdict[filename][catena] = mi
                    first_mi = mi

    for filename in catdict:
        catdict_lists[filename] = list(sorted(catdict[filename].items(), key=lambda x: -x[1]))

    for filename in catdict_lists:
        basename = filename.replace("/", "_")

        logger.info("Catenae in {}: {}".format(filename, len(catdict_lists[filename])))

        with gzip.open(output_dir+"/{}.TOP{}".format(basename, topk), "wt") as fout:
            for catena, mi in catdict_lists[filename][:topk]:
                print("{}\t{}".format(catena, mi), file=fout)

    # SPEARMAN
    stats = {}
    p_values = {}
    keys_list = set()
    zeros = {}
    vectors = {}

    with open(output_dir+"/spearmanr-TOP{}.txt".format(topk), "w") as fout:
        for filename in catdict:

            stats[filename] = collections.defaultdict(lambda: -1.0)
            p_values[filename] = collections.defaultdict(lambda: -1.0)
            keys_list.add(filename)

            dims = catdict_lists[filename][:topk]
            print(filename, file=fout)
            print("\t".join([x[0] for x in dims]), file=fout)

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
    
        for filename in stats:
            for filename2 in stats[filename]:
                s, p_s = stats[filename][filename2], p_values[filename][filename2]
                print("{}\t{}\t{}\t{}".format(filename, filename2, s, p_s), file=fout)
