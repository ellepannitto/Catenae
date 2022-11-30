# pylint: disable=unspecified-encoding
"""_summary_
"""
import logging
import collections
import gzip

from pathlib import Path
from typing import List

import tqdm

from scipy.stats.mstats import spearmanr


logger = logging.getLogger(__name__)


def correlate(output_dir: Path, filenames_list: List[str], # pylint:disable=too-many-locals,too-many-branches
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

            # first_mi = 100
            line = fin.readline()

            for line in fin:
                linesplit = line.strip().split("\t")
                catena = linesplit[0].lower()
                freq = float(linesplit[1])
                mutual_information = float(linesplit[2])
                if freq > fr_threshold:
                    catdict[filename][catena] = mutual_information
                    # first_mi = mutual_information

    for filename in catdict: # pylint:disable=consider-using-dict-items
        catdict_lists[filename] = list(sorted(catdict[filename].items(), key=lambda x: -x[1]))

    for filename, filename_lst in catdict_lists.items():
        basename = filename.replace("/", "_")

        logger.info("Catenae in %s: %d", filename, len(filename_lst))

        with gzip.open(output_dir.joinpath(f"{basename}.TOP{topk}"), "wt") as fout:
            for catena, mutual_information in filename_lst[:topk]:
                print(f"{catena}\t{mutual_information}", file=fout)

    # SPEARMAN
    stats = {}
    p_values = {}
    keys_list = set()
    zeros = {}
    vectors = {}

    with open(output_dir.joinpath("/spearmanr-TOP{topk}.txt"), "w") as fout:

        for fname1, _ in catdict.items():

            stats[fname1] = collections.defaultdict(lambda: -1.0)
            p_values[fname1] = collections.defaultdict(lambda: -1.0)
            keys_list.add(fname1)

            dims = catdict_lists[fname1][:topk]
            #print(filename, file=fout)
            #print("\t".join([x[0] for x in dims]), file=fout)

            for fname2, fname2_lst in catdict.items():
                vectors[fname1] = []
                vectors[fname2] = []
                zeros[fname2] = 0

                for catena, mutual_information in dims:

                    vectors[fname1].append(mutual_information)

                    if catena in fname2_lst:
                        vectors[fname2].append(fname2_lst[catena])
                    else:
                        vectors[fname2].append(0)
                        zeros[fname2] += 1

                stat_value, p_value = spearmanr(vectors[fname1], vectors[fname2])
                stats[fname1][fname2] = stat_value
                p_values[fname1][fname2] = p_value

        for fname1 in stats: # pylint:disable=consider-using-dict-items
            for fname2 in stats[fname1]:
                stat_value, p_value = stats[fname1][fname2], p_values[fname1][fname2]
                print(f"{fname1}\t{fname2}\t{stat_value}\t{p_value}", file=fout)


def corecatenae(output_dir: Path, input_filenames_list: List[Path], # pylint:disable=too-many-locals
                babbling_filenames_list: List[Path], topk: int) -> None:
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
        file_idx = int(fname.parts[-2]) - 1

        with gzip.open(fname, "rt") as fin:
            fin.readline()

            i=0

            for line in tqdm.tqdm(fin):
                line = line.strip().split("\t")

                catena, _, mutual_information = line

                if not catena in babblings:
                    babblings[catena] = [None]*len(babbling_filenames_list)
                    ranks[catena] = [None]*len(babbling_filenames_list)

                babblings[catena][file_idx] = mutual_information
                ranks[catena][file_idx] = i+1

                i+=1
                if i>topk:
                    break


    for fname in input_filenames_list:
        file_idx = int(fname.parts[-2]) - 1

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
        composed_header = "catena\t"

        lst = ["input_freq_"+str(i).zfill(2) for i in range(1,len(input_filenames_list)+1)]
        composed_header+= "\t".join(lst)+"\t"

        lst = ["babbling_mi_"+str(i).zfill(2) for i in range(1,len(input_filenames_list)+1)]
        composed_header+= "\t".join(lst)+"\t"

        lst = ["babbling_rank_"+str(i).zfill(2) for i in range(1,len(input_filenames_list)+1)]
        composed_header+= "\t".join(lst)

        print(composed_header, file=fout)

        for catena in babblings: # pylint:disable=consider-using-dict-items
            lst = [catena]

            for freq in inputs[catena]:
                lst.append(str(freq))

            for mutual_information in babblings[catena]:
                lst.append(str(mutual_information))

            for rank in ranks[catena]:
                lst.append(str(rank))

            print("\t".join(lst), file=fout)

