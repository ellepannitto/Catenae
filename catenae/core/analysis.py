import logging
import collections
import gzip

from pathlib import Path
from typing import List

import tqdm

from scipy.stats.mstats import spearmanr

logger = logging.getLogger(__name__)


def correlate(output_dir: Path, filenames_list: List[str],
              topk: int, mi_threshold: int, fr_threshold: int) -> None:
    """_summary_

    Args:
        output_dir (str): _description_
        filenames_list (List[str]): _description_
        topk (int): _description_
        mi_threshold (int): _description_
        fr_threshold (int): _description_
    """

    catdict = {}
    catdict_lists = {}

    for filename in filenames_list:
        with gzip.open(filename, "rt") as fin:
            catdict[filename] = {}

            first_mi = 100
            line = fin.readline()

            for line in fin:
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

        with gzip.open(output_dir.joinpath(f"{basename}.TOP{topk}"), "wt") as fout:
            for catena, mi in catdict_lists[filename][:topk]:
                print(f"{catena}\t{mi}", file=fout)

    # SPEARMAN
    stats = {}
    p_values = {}
    keys_list = set()
    zeros = {}
    vectors = {}

    with open(output_dir.joinpath("/spearmanr-TOP{topk}.txt"), "w") as fout:
        for filename in catdict:

            stats[filename] = collections.defaultdict(lambda: -1.0)
            p_values[filename] = collections.defaultdict(lambda: -1.0)
            keys_list.add(filename)

            dims = catdict_lists[filename][:topk]
            #print(filename, file=fout)
            #print("\t".join([x[0] for x in dims]), file=fout)

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


def corecatenae(output_dir: Path, input_filenames_list: List[str],
                babbling_filenames_list: List[str], topk: int) -> None:
    """_summary_

    Args:
        output_dir (str): _description_
        input_filenames_list (List[str]): _description_
        babbling_filenames_list (List[str]): _description_
        topk (int): _description_
    """

    inputs = {}
    babblings = {}
    ranks = {}

    for fname in babbling_filenames_list:
        file_idx = int(fname.split("/")[-2]) -1

        with gzip.open(fname, "rt") as fin:
            fin.readline()

            i=0

            for line in tqdm.tqdm(fin):
                line = line.strip().split("\t")

                catena, _, mi = line

                if not catena in babblings:
                    babblings[catena] = [None]*len(babbling_filenames_list)
                    ranks[catena] = [None]*len(babbling_filenames_list)

                babblings[catena][file_idx] = mi
                ranks[catena][file_idx] = i+1

                i+=1
                if i>topk:
                    break


    for fname in input_filenames_list:
        file_idx = int(fname.split("/")[-2]) -1

        with gzip.open(fname, "rt") as fin:
            fin.readline()

            for line in tqdm.tqdm(fin):
                line = line.strip().split("\t")

                catena, frequency, _ = line

                if catena in babblings:
                    if not catena in inputs:
                        inputs[catena] = [None]*len(input_filenames_list)

                    inputs[catena][file_idx] = frequency


    with open(output_dir.joinpath("babblingstats.tsv"), "w") as fout:
        s = "catena\t"
        l = ["input_freq_"+str(i).zfill(2) for i in range(1,11)]
        s+= "\t".join(l)+"\t"

        l = ["babbling_mi_"+str(i).zfill(2) for i in range(1,11)]
        s+= "\t".join(l)+"\t"

        l = ["babbling_rank_"+str(i).zfill(2) for i in range(1,11)]
        s+= "\t".join(l)

        print(s, file=fout)

        for catena in babblings:
            lst = [catena]

            for i, freq in enumerate(inputs[catena]):
                lst.append(str(freq))

            for i, mi in enumerate(babblings[catena]):
                lst.append(str(mi))

            for i, rank in enumerate(ranks[catena]):
                lst.append(str(rank))

            print("\t".join(lst), file=fout)

            # print("\t".join(str(x) for x in lst))
            # print(babblings[catena])
            # print(inputs[catena])
            # input()
