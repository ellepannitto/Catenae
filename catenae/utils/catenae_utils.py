import string
import itertools

from typing import List

ADMITTED_CHARS = string.ascii_letters+".-' "
POS_TO_EXCLUDE = ("PUNCT",)
RELS_TO_EXCLUDE = ("discourse", "fixed", "flat", "comound", "list", "parataxis", "orphan",
                   "goeswith", "reparandum", "punct", "dep")


def word_admitted(word: str, admitted_chars: str = ADMITTED_CHARS) -> bool:
    return all(c in admitted_chars for c in word)


def pos_admitted(pos: str, poslist: List[str] = POS_TO_EXCLUDE) -> bool:
    return pos not in poslist


def rel_admitted(rel: str, relslist: List[str] = RELS_TO_EXCLUDE) -> bool:
    return rel not in relslist


def recursive_catenae_extraction(A, tree_children, min_len_catena, max_len_catena):
    # if A is a leaf
    if A not in tree_children:
        return [[A]], [[A]]
        # return [[A]], []

    else:
        found_catenae = []
        list_of_indep_catenae = [[[A]]]

        for a_child in tree_children[A]:
            c, all_c = recursive_catenae_extraction(a_child, tree_children,
                                                    min_len_catena, max_len_catena)

            found_catenae += all_c
            list_of_indep_catenae.append([[None]] + c)

        X = []
        for tup in itertools.product(*list_of_indep_catenae):
            new_catena = list(sorted(filter(lambda x: x is not None, sum(tup, []))))
            if min_len_catena <= len(new_catena) <= max_len_catena:
                X.append(new_catena)

        return X, X+found_catenae
