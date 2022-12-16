from typing import Iterable, Tuple, List

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
    # sim_dif = cxn1_values ^ cxn2_values

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