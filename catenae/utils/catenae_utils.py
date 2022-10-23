"""
Set of utilities to handle catenae extraction.
"""
import string
import itertools

from typing import List

ADMITTED_CHARS = string.ascii_letters+".-' "
POS_TO_EXCLUDE = ("PUNCT",)
RELS_TO_EXCLUDE = ("discourse", "fixed", "flat", "comound", "list", "parataxis", "orphan",
                   "goeswith", "reparandum", "punct", "dep")


def word_admitted(word: str, admitted_chars: str = ADMITTED_CHARS) -> bool:
    """Check if word contains only admitted characters.

    Args:
        word (str): word to check.
        admitted_chars (str, optional): string containing all admitted chars.
                                        Defaults to ADMITTED_CHARS.

    Returns:
        bool: True is all characters are admitted, False otherwise.
    """
    return all(c in admitted_chars for c in word)


def pos_admitted(pos: str, pos_to_exclude: List[str] = POS_TO_EXCLUDE) -> bool:
    """Check if Part-of-Speech is among the admitted ones

    Args:
        pos (str): Part-of-Speech
        pos_to_exclude (List[str], optional): List of Parts-of-Speech not admitted.
                                              Defaults to POS_TO_EXCLUDE.

    Returns:
        bool: True if pos is not in pos_to_exclude, False otherwise
    """
    return pos not in pos_to_exclude


def rel_admitted(rel: str, rels_to_exclude: List[str] = RELS_TO_EXCLUDE) -> bool:
    """Check if Syntactic Relation is among the admitted ones

    Args:
        rel (str): Syntactic Relation
        rels_to_exclude (List[str], optional): List of Syntactic Relations not admitted.
                                               Defaults to RELS_TO_EXCLUDE.

    Returns:
        bool: True if rel is not in pos_to_exclude, False otherwise
    """
    return rel not in rels_to_exclude


def recursive_catenae_extraction(A, tree_children, min_len_catena, max_len_catena):
    """_summary_

    Args:
        A (_type_): _description_
        tree_children (_type_): _description_
        min_len_catena (_type_): _description_
        max_len_catena (_type_): _description_

    Returns:
        _type_: _description_
    """
    # if A is a leaf
    if A not in tree_children:
        return [[A]], [[A]]
        # return [[A]], []

    # else
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
