# pylint: disable=unspecified-encoding
"""
Functions to compute various kinds of statistics on corpus.
"""
import logging
import collections
import itertools

from typing import Any, List, Tuple
from pathlib import Path

import tqdm

from catenae.utils import corpus_utils as cutils
from catenae.utils import files_utils as futils

logger = logging.getLogger(__name__)


def extract_rel(corpus_filepath: str, synrel: str) -> Tuple[List[Any], List[Any], List[Any]]: # pylint: disable=too-many-locals
    # TODO: check shape of lists
    """_summary_

    Args:
        corpus_filepath (str): _description_
        synrel (str): _description_

    Returns:
        Tuple[List[Any], List[Any], List[Any]]: _description_
    """

    tot_lemmatized = 0
    freqdist_lemmatized = collections.defaultdict(int)

    tot_mixed = 0
    freqdist_mixed = collections.defaultdict(int)

    tot_abstract = 0
    freqdist_abstract = collections.defaultdict(int)

    for sentence in cutils.reader(corpus_filepath):
        tree = collections.defaultdict(int)
        items = []

        for token in sentence:
            offset, _, _, _, _, _, head, rel, _, _ = token
            offset = int(offset)
            head = int(head)
            tree[offset - 1] = head - 1

            if rel == synrel:
                items.append(offset-1)

        for item in items:
            minpos, maxpos = min(item, tree[item]), max(item, tree[item])

            pair_lemma = (sentence[minpos][2], sentence[maxpos][2])
            pair_pos = ("@"+sentence[minpos][3], "@"+sentence[maxpos][3])
            pair_rel = ("_"+sentence[minpos][7], "_"+sentence[maxpos][7])

            pairs_list = [pair_lemma, pair_pos, pair_rel]

            combinations = list(itertools.combinations_with_replacement([0, 1, 2], 2))
            combinations_list = []
            for a, b in combinations: # pylint: disable=invalid-name
                combinations_list.append((pairs_list[a][0], pairs_list[b][1]))
                if a == 0 and b == 0:
                    freqdist_lemmatized[(pairs_list[a][0], pairs_list[b][1])] += 1
                    tot_lemmatized += 1
                elif a == 0 or b == 0:
                    freqdist_mixed[(pairs_list[a][0], pairs_list[b][1])] += 1
                    tot_mixed += 1
                else:
                    freqdist_abstract[(pairs_list[a][0], pairs_list[b][1])] += 1
                    tot_abstract += 1

    sorted_freqdist_lemmatized = sorted(freqdist_lemmatized.items(), key=lambda x: -x[1])
    sorted_freqdist_lemmatized = [(x, y, f"{y * 100 / tot_lemmatized:.10f}%")
                                  for x, y in sorted_freqdist_lemmatized]

    sorted_freqdist_mixed = sorted(freqdist_mixed.items(), key=lambda x: -x[1])
    sorted_freqdist_mixed = [(x, y, f"{y * 100 / tot_mixed:.10f}%")
                             for x, y in sorted_freqdist_mixed]

    sorted_freqdist_abstract = sorted(freqdist_abstract.items(), key=lambda x: -x[1])
    sorted_freqdist_abstract = [(x, y, f"{y * 100 / tot_abstract:.10f}%")
                                for x, y in sorted_freqdist_abstract]

    return sorted_freqdist_lemmatized, sorted_freqdist_mixed, sorted_freqdist_abstract


def extract_edges(corpus_filepath: str, distance: int) -> List[Any]:
    # TODO: check shape of lists
    """_summary_

    Args:
        corpus_filepath (str): _description_
        distance (int): _description_

    Returns:
        List[Any]: _description_
    """

    tot = 0
    freqdist = collections.defaultdict(int)

    for sentence in cutils.reader(corpus_filepath):
        tree = collections.defaultdict(list)
        verbs = []
        for token in sentence:
            offset, _, _, pos, _, _, head, _, _, _ = token
            offset = int(offset)
            head = int(head)
            if pos == "VERB":
                verbs.append(offset-1)
            if not pos == "PUNCT":
                tree[head - 1].append(offset - 1)

        for verb in verbs:
            if len(tree[verb]) == distance:
                freqdist[sentence[verb][2]] += 1
                tot += 1

    sorted_freqdist = sorted(freqdist.items(), key=lambda x: -x[1])
    sorted_freqdist = [(x, y, f"{y * 100 / tot:.10f}%") for x, y in sorted_freqdist]

    return sorted_freqdist


def extract_subj_verb(corpus_filepath: str) -> Tuple[List[Any], List[Any]]: # pylint: disable=too-many-locals
    # TODO: check shape of lists
    """_summary_

    Args:
        corpus_filepath (str): _description_

    Returns:
        Tuple[List[Any], List[Any]]: _description_
    """

    tot_pre = 0
    freqdist_pre = collections.defaultdict(int)
    tot_post = 0
    freqdist_post = collections.defaultdict(int)

    for sentence in cutils.reader(corpus_filepath):
        tree = collections.defaultdict(int)
        sbjs = []
        for token in sentence:
            offset, _, lemma, _, _, _, head, rel, _, _ = token
            offset = int(offset)
            head = int(head)
            if rel == "nsubj":
                sbjs.append(offset - 1)
                tree[offset-1] = head - 1

        for sbj in sbjs:
            head = tree[sbj]
            distance = sbj - head

            head = sentence[head]
            _, _, lemma, _, _, _, _, _, _, _ = head

            if distance > 0:
                tot_post += 1
                freqdist_post[lemma] += 1
            else:
                tot_pre += 1
                freqdist_pre[lemma] += 1

    sorted_freqdist_pre = sorted(freqdist_pre.items(), key=lambda x: -x[1])
    sorted_freqdist_pre = [(x, y, f"{y * 100 / tot_pre:.10f}%")
                           for x, y in sorted_freqdist_pre]

    sorted_freqdist_post = sorted(freqdist_post.items(), key=lambda x: -x[1])
    sorted_freqdist_post = [(x, y, f"{y * 100 / tot_post:.10f}%")
                            for x, y in sorted_freqdist_post]

    return sorted_freqdist_pre, sorted_freqdist_post


def extract_subj_noun(corpus_filepath: str) -> Tuple[List[Any], List[Any]]: # pylint: disable=too-many-locals
    # TODO: check shape of lists
    """_summary_

    Args:
        corpus_filepath (str): _description_

    Returns:
        Tuple[List[Any], List[Any]]: _description_
    """

    tot_pre = 0
    freqdist_pre = collections.defaultdict(int)
    tot_post = 0
    freqdist_post = collections.defaultdict(int)

    for sentence in cutils.reader(corpus_filepath):
        tree = collections.defaultdict(int)
        sbjs = []
        for token in sentence:
            offset, _, lemma, _, _, _, head, rel, _, _ = token
            offset = int(offset)
            head = int(head)
            if rel == "nsubj":
                sbjs.append(offset - 1)
                tree[offset-1] = head - 1

        for sbj in sbjs:
            head = tree[sbj]
            distance = sbj - head

            noun = sentence[sbj]
            _, _, lemma, _, _, _, _, _, _, _ = noun

            if distance > 0:
                tot_post += 1
                freqdist_post[lemma] += 1
            else:
                tot_pre += 1
                freqdist_pre[lemma] += 1

    sorted_freqdist_pre = sorted(freqdist_pre.items(), key=lambda x: -x[1])
    sorted_freqdist_pre = [(x, y, f"{y * 100 / tot_pre:.10f}%") for x, y in sorted_freqdist_pre]

    sorted_freqdist_post = sorted(freqdist_post.items(), key=lambda x: -x[1])
    sorted_freqdist_post = [(x, y, f"{y * 100 / tot_post:.10f}%") for x, y in sorted_freqdist_post]

    return sorted_freqdist_pre, sorted_freqdist_post


def extract_pos(corpus_filepath: str, pos: str) -> List[Any]:
    # TODO: check shape of lists
    """_summary_

    Args:
        corpus_filepath (str): _description_
        pos (str): _description_

    Returns:
        List[Any]: _description_
    """

    tot = 0
    freqdist = collections.defaultdict(int)

    for sentence in cutils.reader(corpus_filepath):
        for token in sentence:
            token_pos = token[3]

            if token_pos == pos:
                freqdist[token[2]] += 1
                tot += 1

    sorted_freqdist = sorted(freqdist.items(), key=lambda x: -x[1])
    sorted_freqdist = [(x, y, f"{y*100/tot:.10f}%") for x, y in sorted_freqdist]

    return sorted_freqdist


def extract_morph(corpus_filepath: str, morph_trait: str, morph_value: str) -> List[Any]:
    # TODO: check shape of lists
    """_summary_

    Args:
        corpus_filepath (str): _description_
        morph_trait (str): _description_
        morph_value (str): _description_

    Returns:
        List[Any]: _description_
    """

    tot = 0
    freqdist = collections.defaultdict(int)
    for sentence in cutils.reader(corpus_filepath):
        for token in sentence:
            token_pos = token[3]
            token_morph = token[5].split("|")

            if token_pos == "VERB" and len(token_morph) > 0:
                for trait in token_morph:
                    name, value = trait.split("=")
                    if name == morph_trait and value == morph_value:
                        freqdist[token[2]] += 1
                        tot += 1

    sorted_freqdist = sorted(freqdist.items(), key=lambda x: -x[1])
    sorted_freqdist = [(x, y, f"{y*100/tot:.10f}%") for x, y in sorted_freqdist]

    return sorted_freqdist


def extract_mood(corpus_filepath: str, mood: str) -> List[Any]:
    # TODO: check shape of lists
    """_summary_

    Args:
        corpus_filepath (str): _description_
        mood (str): _description_

    Returns:
        List[Any]: _description_
    """
    tot = 0
    freqdist = collections.defaultdict(int)
    for sentence in cutils.reader(corpus_filepath): # pylint: disable=too-many-nested-blocks
        for token in sentence:
            token_pos = token[3]
            if token_pos == "VERB":
                token_morph = token[5].split("|")
                if len(token_morph) > 0:
                    for trait in token_morph:
                        name, value = trait.split("=")
                        if name == "Mood" and value == mood:
                            freqdist[token[2]] += 1
                            tot += 1

    sorted_freqdist = sorted(freqdist.items(), key=lambda x: -x[1])
    sorted_freqdist = [(x, y, f"{y*100/tot:.10f}%") for x, y in sorted_freqdist]

    return sorted_freqdist


def extract_form(corpus_filepath: str, form: str) -> List[Any]:
    # TODO: check shape of lists
    """_summary_

    Args:
        corpus_filepath (str): _description_
        form (str): _description_

    Returns:
        List[Any]: _description_
    """

    tot = 0
    freqdist = collections.defaultdict(int)
    for sentence in cutils.reader(corpus_filepath): # pylint: disable=too-many-nested-blocks
        for token in sentence:
            token_pos = token[3]
            if token_pos == "VERB":
                token_morph = token[5].split("|")
                if len(token_morph) > 0:
                    for trait in token_morph:
                        name, value = trait.split("=")
                        if name == "VerbForm" and value == form:
                            freqdist[token[2]] += 1
                            tot += 1
    sorted_freqdist = sorted(freqdist.items(), key=lambda x: -x[1])
    sorted_freqdist = [(x, y, f"{y*100/tot:.10f}%") for x, y in sorted_freqdist]

    return sorted_freqdist


def compute_pos_distribution(output_dir: Path, corpus_dirpath: Path,
                             pos_list: List[str]) -> None:
    """_summary_

    Args:
        output_dir (Path): _description_
        corpus_dirpath (Path): _description_
        pos_list (List[str]): _description_
    """

    for pos in tqdm.tqdm(pos_list):
        filenames_it = tqdm.tqdm(corpus_dirpath.iterdir())
        for filename in filenames_it:
            if filename.suffix == ".conll":
                filenames_it.set_description(filename)
                sorted_freqs = extract_pos(corpus_dirpath.joinpath(filename), pos)
                corpus_dirpath_parts = corpus_dirpath.parts
                file_basename = filename.stem
                with open(output_dir / f"{corpus_dirpath_parts[-2]}.{file_basename}.{pos}",
                          "w") as fout:
                    futils.print_formatted(sorted_freqs, fout)


def compute_morph_distribution(output_dir: Path, corpus_dirpath: Path,
                               trait: str, values_list: List[str]) -> None:
    """_summary_

    Args:
        output_dir (Path): _description_
        corpus_dirpath (Path): _description_
        trait (str): _description_
        values_list (List[str]): _description_
    """

    for value in tqdm.tqdm(values_list):
        filenames_it = tqdm.tqdm(corpus_dirpath.iterdir())
        for filename in filenames_it:
            if filename.suffix == ".conll":
                filenames_it.set_description(filename)
                sorted_freqs = extract_morph(corpus_dirpath.joinpath(filename), trait, value)

                corpus_dirpath_parts = corpus_dirpath.parts
                file_basename = filename.stem
                with open(output_dir / f"{corpus_dirpath_parts[-2]}.{file_basename}.{trait}_{value}", # pylint: disable=line-too-long
                          "w") as fout:
                    futils.print_formatted(sorted_freqs, fout)


def compute_verbedges_distribution(output_dir: Path, corpus_dirpath: Path,
                                   number_edges: int) -> None:
    """_summary_

    Args:
        output_dir (Path): _description_
        corpus_dirpath (Path): _description_
        number_edges (int): _description_
    """

    filenames_it = tqdm.tqdm(corpus_dirpath.iterdir())
    for filename in filenames_it:
        if filename.suffix == ".conll":
            filenames_it.set_description(filename)

            sorted_freqs = extract_edges(corpus_dirpath.joinpath(filename), number_edges)

            corpus_dirpath_parts = corpus_dirpath.parts
            file_basename = filename.stem
            with open(output_dir / f"{corpus_dirpath_parts[-2]}.{file_basename}.{number_edges}edge",
                      "w") as fout:
                futils.print_formatted(sorted_freqs, fout)


def compute_sbj_distribution(output_dir: Path, corpus_dirpath: Path) -> None:
    """_summary_

    Args:
        output_dir (Path): _description_
        corpus_dirpath (Path): _description_
    """

    filenames_it = tqdm.tqdm(corpus_dirpath.iterdir())
    for filename in filenames_it:
        if filename.suffix == ".conll":
            filenames_it.set_description(filename)

            sorted_freqs_pre,\
            sorted_freqs_post = extract_subj_verb(corpus_dirpath.joinpath(filename))

            corpus_dirpath_parts = corpus_dirpath.parts
            file_basename = filename.stem

            with open(output_dir / f"{corpus_dirpath_parts[-2]}.{file_basename}.presubjverb",
                      "w") as fout_pre, \
                open(output_dir / f"{corpus_dirpath_parts[-2]}.{file_basename}.postsubjverb",
                     "w") as fout_post:
                futils.print_formatted(sorted_freqs_pre, fout_pre)
                futils.print_formatted(sorted_freqs_post, fout_post)

            sorted_freqs_pre,\
            sorted_freqs_post = extract_subj_noun(corpus_dirpath.joinpath(filename))

            with open(output_dir / f"{corpus_dirpath_parts[-2]}.{file_basename}.presubjnoun",
                      "w") as fout_pre, \
                open(output_dir / f"{corpus_dirpath_parts[-2]}.{file_basename}.postsubjnoun",
                     "w") as fout_post:
                futils.print_formatted(sorted_freqs_pre, fout_pre)
                futils.print_formatted(sorted_freqs_post, fout_post)


def compute_synrel_distribution(output_dir: Path, corpus_dirpath: Path,
                                synrel_list: List[str]) -> None:
    """_summary_

    Args:
        output_dir (Path): _description_
        corpus_dirpath (Path): _description_
        synrel_list (List[str]): _description_
    """

    for rel in tqdm.tqdm(synrel_list):
        filenames_it = tqdm.tqdm(corpus_dirpath.iterdir())
        for filename in filenames_it:
            if filename.suffix == ".conll":
                filenames_it.set_description(filename)

                sorted_freqs_lem,\
                sorted_freqs_mix,\
                sorted_freqs_abs = extract_rel(corpus_dirpath.joinpath(filename), rel)

                corpus_dirpath_parts = corpus_dirpath.parts
                file_basename = filename.stem

                with open(output_dir / f"{corpus_dirpath_parts[-2]}.{file_basename}.lem_{rel}",
                          "w") as fout_lem, \
                     open(output_dir / f"{corpus_dirpath_parts[-2]}.{file_basename}.mix_{rel}",
                          "w") as fout_mix, \
                     open(output_dir / f"{corpus_dirpath_parts[-2]}.{file_basename}.abs_{rel}",
                          "w") as fout_abs:

                    futils.print_formatted(sorted_freqs_lem, fout_lem)
                    futils.print_formatted(sorted_freqs_mix, fout_mix)
                    futils.print_formatted(sorted_freqs_abs, fout_abs)
