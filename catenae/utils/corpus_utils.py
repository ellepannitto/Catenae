# pylint: disable=unspecified-encoding
"""
Set of utilities for reading Corpus.
"""
from typing import List

def plain_conll_reader(filepath: str, min_len: int = 0, max_len: int = 300) -> List[str]:
    """Read through CoNLL formatted file.

    Args:
        filepath (str): path to input file
        min_len (int, optional): Minimum length for sentence to be considered. Defaults to 0.
        max_len (int, optional): Maximum length for sentence to be considered. Defaults to 300.

    Yields:
        List[str]: Sentence represented as list of strings, one for each token.
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
    """Renders a sentence in linear format (list of its token forms).

    Args:
        sentence (List[str]): List of CoNLL tokens contained in sentence.

    Returns:
        str: Linearized sentence (sequence of its token forms).
    """
    res = []
    for token in sentence:
        res.append(token.split("\t")[1])
    return " ".join(res)


def reader(input_file: str) -> List[List[str]]: # TODO: check docstring
    """Read input file containing ProfilingUD output.

    Args:
        input_file (str): path to input file

    Yields:
        Iterator[List[List[str]]]: list representing sentence
    """
    sentence = []
    with open(input_file) as fin:
        for line in fin:
            linestrip = line.strip()
            if len(linestrip) and not linestrip[0] == "#":
                linesplit = linestrip.split("\t")
                sentence.append(linesplit)
            else:
                if len(sentence) > 1:
                    yield sentence
                sentence = []
    if len(sentence) > 1:
        yield sentence
