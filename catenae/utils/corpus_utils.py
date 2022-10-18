"""
Set of utilities for reading Corpus.
"""
from typing import List

def plain_conll_reader(filepath: str, min_len: int = 0, max_len: int = 300) -> List[str]:
    """_summary_

    Args:
        filepath (str): _description_
        min_len (int, optional): _description_. Defaults to 0.
        max_len (int, optional): _description_. Defaults to 300.

    Yields:
        List[str]: _description_
    """
    with open(filepath) as fin:
        sentence = []
        to_include = True
        for line in fin:
            line = line.strip()
            if line.startswith("#") or len(line) == 0:
                if min_len < len(sentence) < max_len:
                    if to_include:
                        yield sentence
                sentence = []
                to_include = True
                if line.startswith("speaker: CHI"):
                    to_include = False
            else:
                line = line.strip()
                if len(line):
                    sentence.append(line)

        if min_len < len(sentence) < max_len and to_include:
            yield sentence


def get_linear(sentence: List[str]) -> str:
    """_summary_

    Args:
        sentence (List[str]): _description_

    Returns:
        str: _description_
    """
    res = []
    for token in sentence:
        res.append(token.split("\t")[1])
    return " ".join(res)
