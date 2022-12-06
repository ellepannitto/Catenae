# pylint: disable=unspecified-encoding
import logging
import itertools
import math
import functools
import collections

from pathlib import Path
from multiprocessing import Pool

from typing import Iterable, Tuple, List, Dict

import tqdm
from tqdm.contrib.concurrent import process_map
import networkx as nx

from catenae.utils import corpus_utils as cutils
from catenae.utils import data_utils as dutils
from catenae.utils import catenae_utils as catutils


logger = logging.getLogger(__name__)


def compute_matrix(output_dir: Path, input_dir: Path, catenae_fpath: Path,
                   min_len_catena: int, max_len_catena: int,
                   multiprocess: bool, n_workers: int) -> None:
    """_summary_

    Args:
        output_dir (Path): _description_
        input_dir (Path): _description_
        catenae_fpath (Path): _description_
        min_len_catena (int): _description_
        max_len_catena (int): _description_
        multiprocess (bool): _description_
        n_workers (int): _description_
    """

    # TODO implement multiprocess

    catenae_glass = dutils.load_catenae_set(catenae_fpath, math.inf)

    input_files_it = tqdm.tqdm(input_dir.iterdir())

    for input_file in input_files_it:

        with open(output_dir.joinpath(input_file.name), "w") as fout:
            input_files_it.set_description(f"Reading file {input_file}")

            for _, sentence in enumerate(cutils.plain_conll_reader(input_file.absolute(),
                                                                   min_len=1, max_len=25)):
                if sentence:
                    children = {}
                    tokens = {}
                    postags = {}
                    rels = {}
                    tokens_to_remove = []

                    projected_sentence = [dutils.DefaultList([el.split("\t")[1]], "_")
                                          for el in sentence]
                    currently_looking_at = 1

                    for token in sentence:
                        token = token.split("\t")

                        position, word, _, pos, _, _, head, rel, _, _ = token
                        if catutils.pos_admitted(pos) and catutils.rel_admitted(rel):

                            position = int(position)

                            if not catutils.word_admitted(word):
                                tokens_to_remove.append(position)

                            head = int(head)
                            if head not in children:
                                children[head] = []

                            children[head].append(position)
                            tokens[position] = word
                            postags[position] = "_"+pos
                            rels[position] = "@"+rel

                    if 0 in children:
                        root = children[0][0]
                        _, catenae = catutils.recursive_catenae_extraction(root, children,
                                                                           min_len_catena,
                                                                           max_len_catena)

                        explicit_catenae = set()

                        for catena in catenae:

                            if all(x not in tokens_to_remove for x in catena):
                                tokensandpostags = [[tokens[x] for x in catena],
                                                    [postags[x] for x in catena],
                                                    [rels[x] for x in catena]]

                                temp = [(0, 1, 2)] * len(catena)
                                X = list(itertools.product(*temp))

                                for c in X:
                                    cat = []
                                    for i, el in enumerate(c):
                                        cat.append(tokensandpostags[el][i])
                                    cat = tuple(cat)

                                    explicit_catenae.add(cat)

                                    cat_to_look_for = "|".join(cat)

                                    if cat_to_look_for in catenae_glass:

                                        for i, label in zip(catena, cat):
                                            projected_sentence[i-1][currently_looking_at] = label
                                        currently_looking_at += 1

                        for lst in projected_sentence:
                            lst.fill(currently_looking_at)

                        print("\n".join("\t".join(x) for x in projected_sentence), file=fout)
                        print("\n", file=fout)


def matchable(cxn1: Iterable[str], cxn2: Iterable[str]) -> bool:
    """_summary_

    Args:
        cxn1 (Iterable[str]): _description_
        cxn2 (Iterable[str]): _description_

    Returns:
        bool: _description_
    """

    cxn1_values = set(i for i, el in enumerate(cxn1) if not el == "_")
    cxn2_values = set(i for i, el in enumerate(cxn2) if not el == "_")

    intersection = cxn1_values & cxn2_values
    sim_dif = cxn1_values ^ cxn2_values

    return all(cxn1[i]==cxn2[i] for i in intersection) and \
        not (cxn1_values < cxn2_values or cxn2_values < cxn1_values)


def update(sentence: Iterable[str], cxn: Iterable[str]) -> Tuple[Iterable[str], int]:
    ret = [x for x in sentence]
    added = 0
    for i, cxn_el in enumerate(cxn):
        if not cxn_el == "_":
            ret[i] = cxn_el
            added += 1

    return ret, added


def process(search_space, len_sentence):

    solutions = collections.defaultdict(list)

    G = nx.Graph()
    for i, _ in search_space.items():
        G.add_node(i)

    for i, cxn1 in search_space.items():
        for j, cxn2 in search_space.items():
            if j>i:
                # print("\t\t".join(cxn1))
                # print("\t\t".join(cxn2))
                # print(matchable(cxn1, cxn2))
                if matchable(cxn1, cxn2):
                    G.add_edge(i, j)
                # input()

    # G = nx.Graph()
    # for i, cxn in enumerate(search_space):
    #     G.add_node(i)

    # for i, cxn1 in enumerate(search_space):
    #     for j, cxn2 in enumerate(search_space):
    #         if matchable(cxn1, cxn2):
    #             G.add_edge(i, j)

    for element in nx.find_cliques(G):
        projected_sentence = ["_"]*len_sentence

        for idx in element:
            projected_sentence, _ = update(projected_sentence, search_space[idx])

        solutions[tuple(projected_sentence)].append(element)

    return solutions


def add_lexical_items(sentence: Iterable[str], collapsed_sentence: Iterable[str]) -> Iterable[str]:
    ret = [x for x in collapsed_sentence]

    for i, lem in enumerate(sentence):
        if ret[i] == "_":
            ret[i] = f"+{lem}"

    return ret


def lenscore(x):
    return len([y for y in x if not y == "_"])


def abscore(x):
    new_x = [y for y in x if not y == "_"]

    rels = [z for z in new_x if z[0]=="@"]
    pos = [z for z in new_x if z[0]=="_"]

    n_rels = len(rels)
    n_pos = len(pos)
    n_words = len(new_x) - len(rels) - len(pos)

    return (3*n_words + 2*n_pos + n_rels) / len(new_x)


def collapse_matrix(output_dir: Path, input_dir: Path, catenae_fpath: Path,
                    multiprocessing: bool, n_workers: int, chunksize: int) -> None:
    """_summary_

    Args:
        output_dir (Path): _description_
        input_dir (Path): _description_
        catenae_fpath (Path): _description_
        multiprocessing (bool): _description_
        n_workers (int): _description_
        chunksize (int): _description_
    """

    catenae_idx = {cxn:i for i, cxn in enumerate(dutils.load_catenae_set(catenae_fpath, math.inf))}
    with open(output_dir.joinpath("catenae.idx"), "w") as fout_catenae:
        for cxn, idx in catenae_idx.items():
            print(f"{idx}\t{cxn}", file=fout_catenae)

    filenames = input_dir.iterdir()

    if multiprocessing:
        process_map(functools.partial(collapse, output_dir=output_dir,
                                      input_dir=input_dir, catenae_idx=catenae_idx),
                    filenames, max_workers=n_workers, chunksize=chunksize)

        # with Pool(10) as p:
        #     ret = list(tqdm.tqdm(p.imap(functools.partial(basic_collapse, output_dir=output_dir, input_dir=input_dir),
        #                          filenames)))

    else:
        lstdir_it = tqdm.tqdm(filenames)
        for filename in lstdir_it:
            filename = filename.name
            lstdir_it.set_description(f"Processing {filename}")
            collapse(filename, output_dir, input_dir, catenae_idx)


def not_underscore(lst: List[str]) -> int:
    """_summary_

    Args:
        lst (List[str]): _description_

    Returns:
        int: _description_
    """
    n = 0
    for el in lst:
        if not el == "_":
            n+=1
    return n


def strip_underscore(cxn: List[str]) -> str:
    """_summary_

    Args:
        cxn (List[str]): _description_

    Returns:
        str: _description_
    """

    # print(cxn)

    new_cxn = [x for x in cxn if not x == "_"]
    # print("|".join(new_cxn))
    return "|".join(new_cxn)


def collapse(filename: Path, output_dir: Path, input_dir: Path,
             catenae_idx: Dict[int, str]) -> None:
    """_summary_

    Args:
        filename (Path): _description_
        output_dir (Path): _description_
        input_dir (Path): _description_
        catenae_fpath (Path): _description_
    """



    with open(input_dir.joinpath(filename)) as fin, \
        open(output_dir.joinpath(filename), "w") as fout_translation, \
            open(output_dir.joinpath(f"{filename}.cxns"), "w") as fout_cxns:

        fin_lines = fin.readlines()
        mat = []
        # lines_it = tqdm.tqdm(fin_lines)
        # lines_it.set_description(filename)
        for line in tqdm.tqdm(fin_lines):
            line = line.strip()

            if line:
                mat.append(line.split("\t"))
            else:
                if mat:
                    # print("\n".join("\t".join(el) for el in mat))
                    rev_mat = [[x[i] for x in mat] for i in range(len(mat[0]))]
#                    if len(rev_mat) > 1 and len(rev_mat) < 200: #TODO: check here

                    if len(rev_mat) > 1:
                        full_sentence, search_space = rev_mat[0], rev_mat[1:]
                        len_sentence = len(full_sentence)
                        # search_space = [(not_underscore(el), el) for el in search_space]

                        # search_space.sort(key=operator.itemgetter(0))

                        # groups = itertools.groupby(search_space, key=operator.itemgetter(0))

                        # search_space = [[item for item in data] for (key, data) in groups]

                        search_space_dict = collections.defaultdict(list)
                        for cxn in search_space:
                            search_space_dict[catenae_idx[strip_underscore(cxn)]].append(cxn)
                        # search_space = {catenae_idx[strip_underscore(cxn)]: cxn
                        #                 for cxn in search_space}

                        search_space_uniquedict = {}
                        for i, el_lst in search_space_dict.items():
                            if len(el_lst) == 1:
                                search_space_uniquedict[str(i)] = el_lst[0]
                            else:
                                idx = 1
                                for el in el_lst:
                                    search_space_uniquedict[f"{str(i)}_{idx}"] = el
                                    idx += 1


                        # for el in search_space_uniquedict:
                        #     print(el, search_space_uniquedict[el])
                        # input()

                        solutions = process(search_space_uniquedict, len_sentence)



                        # print(solutions)

                        # scored_solutions = [(x, y, abscore(y), lenscore(y)) for x, y in solutions]
                        # sorted_solutions = sorted(scored_solutions, key=lambda x: (x[2], x[3]), reverse=True)

                        print("SENTENCE:", " ".join(full_sentence), file=fout_translation)
                        print("SENTENCE:", " ".join(full_sentence), file=fout_cxns)
                        # print("SENTENCE:", " ".join(rev_mat[0]))
                        # print("SENTENCE:", " ".join(rev_mat[0]))

                        print("\t".join(search_space_uniquedict.keys()), file=fout_cxns)

                        for solution, built_from_lst in solutions.items():
                            solution = add_lexical_items(full_sentence, list(solution))
                            to_print = " ".join(solution)

                            for built_from in built_from_lst:
                                built_from_str = " ".join(built_from)
                                to_print += f"\t{built_from_str}"

                            print(to_print, file=fout_translation)
                            # print(" ".join(solution),
                            #       "\t", built_from_str, file=fout_translation)
                        print("\n", file=fout_cxns)
                        print("\n", file=fout_translation)

                        # for built_from, solution, abscore_v, lenscore_v in sorted_solutions:
                        #     built_from_str = " ".join(str(x) for x in built_from)

                        #     solution = add_lexical_items(full_sentence, list(solution))
                        #     print("\t", f"{abscore_v:.2f}", "\t", lenscore_v,
                        #         "\t\t", " ".join(solution),
                        #         "\t", built_from_str, file=fout_translation)
                            # print("\t", f"{abscore_v:.2f}", "\t", lenscore_v,
                                # "\t\t", " ".join(solution),
                                # "\t", built_from_str)
                mat = []


# TODO: switch to tqdm multiprocess