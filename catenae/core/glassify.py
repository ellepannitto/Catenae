import logging
import glob
import itertools
import math
import os

from pathlib import Path

import tqdm

from catenae.utils import corpus_utils as cutils
from catenae.utils import data_utils as dutils
from catenae.utils import catenae_utils as catutils


logger = logging.getLogger(__name__)


def compute_matrix(output_dir: Path, input_dir: str, catenae_fpath: str,
                   min_len_catena: int, max_len_catena: int,
                   multiprocess: bool = False) -> None:

    # TODO implement multiprocess

    catenae_glass = dutils.load_catenae_set(catenae_fpath, math.inf)

    input_files = glob.glob(input_dir+"/*")
    input_files_it = tqdm.tqdm(input_files)

    for input_file in input_files_it:
        with open(output_dir.joinpath(os.path.basename(input_file)), "w") as fout:

            input_files_it.set_description(f"Reading file {input_file}")

            for _, sentence in enumerate(cutils.plain_conll_reader(input_file,
                                                                   min_len=1, max_len=25)):

                if sentence:
                    children = {}
                    tokens = {}
                    postags = {}
                    rels = {}
                    tokens_to_remove = []

                    projected_sentence = [dutils.DefaultList([el.split("\t")[1]], "_") for el in sentence]
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
                        _, catenae = catutils.recursive_catenae_extraction(root, children, min_len_catena, max_len_catena)

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


def collapse_matrix(output_dir: Path, input_dir: str):

    for filename in os.listdir(input_dir):
        with open(input_dir+"/"+filename) as fin:
            mat = []
            for line in fin:
                line = line.strip()

                if line:
                    mat.append(line.split("\t"))
                else:
                    if mat:
                        # process(mat)
                        process_bruteforce(mat)
                    mat = []


def matchable(sentence, candidate_cxn):
    # print("matching", sentence, "with", candidate_cxn)
    matchable = True
    for i, sent_el in enumerate(sentence):
        cxn_el = candidate_cxn[i]



        if not (sent_el == "_" or sent_el == cxn_el):
            matchable = False

        # if not cxn_el == "_" and not sent_el == "_":
        #     matchable = False

    return matchable


def update(sentence, cxn):
    ret = ["_"]*len(sentence)
    added = 0
    for i, cxn_el in enumerate(cxn):
        if not cxn_el == "_":
            ret[i] = cxn_el
            added += 1

    return ret, added


def process_bruteforce(mat):

    rev_mat = [[x[i] for x in mat] for i in range(len(mat[0]))]

    if len(rev_mat) > 1:
        search_space = rev_mat[1:]
        projected_sentence = ["_"]*len(mat)
        solutions = rec_find(projected_sentence, search_space, [])
        print("TRANSLATED --- ", " ".join(rev_mat[0]))
        print("INTO --------- ")
        for solution in solutions:
            # print("\t", "\t\t".join(solution))
            print(solution)
        input()


def process(mat):

    rev_mat = [[x[i] for x in mat] for i in range(len(mat[0]))]

    if len(rev_mat) > 1:
        search_space = rev_mat[1:]

        sorted_search_space = sorted(search_space, key=lambda x: abscore(x), reverse=True)

        # sorted_search_space = sorted(sorted_search_space, key=lambda x: lenscore(x), reverse=True)
        sorted_search_space = sorted(sorted_search_space, key=lambda x: lenscore(x))

        current_idx = 0
        something_to_add = True
        projected_sentence = ["_"]*len(mat)
        tot_items = 0
        n_cxns = 0

        while something_to_add and current_idx < len(sorted_search_space):
            candidate = sorted_search_space[current_idx]
            # print("current mat", projected_sentence)
            # print("considering candidate", candidate)
            if matchable(projected_sentence, candidate):
                projected_sentence, added_items = update(projected_sentence, candidate)
                tot_items += added_items
                n_cxns += 1
                # print("cxn is matchable")

            # else:
                # print("NOT MATCHABLE")

            if tot_items == len(projected_sentence):
                something_to_add = False

            current_idx += 1

        print("TRANSLATED --- ", " ".join(rev_mat[0]))
        print("INTO --------- ", " ".join(projected_sentence))
        print("CXNs EMPLOYED: ", n_cxns)
        input()


def lenscore(x):
    return len([y for y in x if not y == "_"])

def abscore(x):
    new_x = [y for y in x if not y == "_"]

    rels = [z for z in new_x if z[0]=="@"]
    pos = [z for z in new_x if z[0]=="_"]

    n_rels = len(rels)
    n_pos = len(pos)
    n_words = len(new_x) - len(rels) - len(pos)

    return 3*n_words + 2*n_pos + n_rels


def rec_find(partial_sentence, search_space, solutions, rec_level):

    print("REC LEVEL:", rec_level)
    print("Evaluating", search_space)
    print("Against sentence", partial_sentence)
    print("Solutions so far:", solutions)


    for i, candidate in enumerate(search_space):

        if matchable(partial_sentence, candidate):

            updated_sentence, _ = update(partial_sentence, candidate)
            updated_search_space = [search_space[k] for k in range(len(search_space)) if not k == i]
            updated_solutions = solutions
            rec_find(updated_sentence, updated_search_space, updated_solutions, rec_level+1)
            solutions.append((updated_sentence, candidate))


    return solutions


if __name__ == "__main__":

    search_space = [["a", "_", "_", "_"],
                    ["a", "b", "_", "_"],
                    ["A", "_", "b", "_"],
                    ["A", "B", "b", "c"]]

    rec_find(["_"]*4, search_space, [], 0)
