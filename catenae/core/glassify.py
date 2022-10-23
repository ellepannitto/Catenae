import logging
import glob
import itertools
import math
import os

from pathlib import Path

import tqdm
import networkx as nx

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
    ret = [x for x in sentence]
    added = 0
    for i, cxn_el in enumerate(cxn):
        if not cxn_el == "_":
            ret[i] = cxn_el
            added += 1

    return ret, added


def process(mat):

    rev_mat = [[x[i] for x in mat] for i in range(len(mat[0]))]
    solutions = []

    if len(rev_mat) > 1:
        search_space = rev_mat[1:]

        G = nx.Graph()
        for i, cxn in enumerate(search_space):
            G.add_node(i)

        for i, cxn1 in enumerate(search_space):
            for j, cxn2 in enumerate(search_space):
                if matchable(cxn1, cxn2):
                    G.add_edge(i, j)

        for element in nx.find_cliques(G):
            projected_sentence = ["_"]*len(rev_mat[0])
            built_from = []

            for idx in element:
                projected_sentence, _ = update(projected_sentence, search_space[idx])
                built_from.append(search_space[idx])

            solutions.append((len(built_from), projected_sentence))

    return rev_mat[0], solutions


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


def collapse_matrix(output_dir: Path, input_dir: str):

    lstdir_it = tqdm.tqdm(os.listdir(input_dir))
    for filename in lstdir_it:
        lstdir_it.set_description(f"Processing {filename}")
        with open(input_dir.joinpath(filename)) as fin, \
            open(output_dir.joinpath(filename), "w") as fout:
            mat = []
            for line in fin:
                line = line.strip()

                if line:
                    mat.append(line.split("\t"))
                else:
                    if mat:
                        original_sentence, solutions = process(mat)
                        scored_solutions = [(x, y, abscore(y), lenscore(y)) for x, y in solutions]
                        sorted_solutions = sorted(scored_solutions, key=lambda x: (x[2], x[3], x[0]), reverse=True)

                        print("SENTENCE:", " ".join(original_sentence), file=fout)
                        print("TRANSLATIONS:", file=fout)
                        for n_cxns, solution, abscore_v, lenscore_v in sorted_solutions:
                            print("\t", n_cxns, "\t", "{:.2f}".format(abscore_v), "\t", lenscore_v, "\t", " ".join(solution), file=fout)

                    mat = []