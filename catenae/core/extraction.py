# pylint: disable=unspecified-encoding
import logging
import collections
import itertools
import uuid
import gzip
import math

from pathlib import Path
from typing import List

import filemerger as fmerger
import tqdm

from catenae.utils import files_utils as futils
from catenae.utils import data_utils as dutils
from catenae.utils import corpus_utils as cutils
from catenae.utils import catenae_utils as catutils


logger = logging.getLogger(__name__)


def process_sentence(sentence, freqdict, catdict, totalsdict,
                     min_len_catena, max_len_catena):

    children = {}
    tokens = {}
    postags = {}
    rels = {}
    tokens_to_remove = []

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

            freqdict[word] += 1
            freqdict["_"+pos] += 1
            freqdict["@"+rel] += 1
            totalsdict["WORDS"] += 1

    if 0 in children:
        root = children[0][0]
        _, catenae = catutils.recursive_catenae_extraction(root, children, min_len_catena, max_len_catena)

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

                    catdict["|".join(cat)] += 1
                    totalsdict[(len(cat))] += 1


def process_cooccurrences(sentence, coocc_dict, catenae_freq,
                          accepted_catenae, min_len_catena, max_len_catena):

    children = {}
    tokens = {}
    postags = {}
    rels = {}
    tokens_to_remove = []

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
                                                           min_len_catena, max_len_catena)

        explicit_catenae = []

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
                    explicit_catenae.append("|".join(cat))

        explicit_catenae = list(filter(lambda x: x in accepted_catenae, explicit_catenae))

        for str_cat_i, str_cat_j in itertools.combinations(explicit_catenae, 2):
            str_cat_i, str_cat_j = min(str_cat_i, str_cat_j), max(str_cat_i, str_cat_j)
            coocc_dict[(str_cat_i, str_cat_j)] += 1
            catenae_freq[str_cat_i] += 1
            catenae_freq[str_cat_j] += 1


def extract_coccurrences(output_dir: Path, input_dir: Path, accepted_catenae_filepath: Path,
                         top_k: int, min_len_sentence: int, max_len_sentence: int,
                         sentences_batch_size: int, min_freq: int,
                         min_len_catena: int, max_len_catena: int,
                         include_words: bool, words_filepath: Path) -> None:

    # TODO: add parameter for top K
    accepted_catenae = dutils.load_catenae_set(accepted_catenae_filepath, top_k)

    if include_words:
        accepted_catenae = dutils.load_catenae_set(words_filepath, float('inf'), accepted_catenae)

    filenames = futils.get_filenames(input_dir)
    total_freqs_global = 0

    for filename in filenames:
        logger.info("Processing file %s", filename)
        iterator = dutils.grouper(cutils.plain_conll_reader(filename, min_len_sentence,
                                                            max_len_sentence),
                                  sentences_batch_size)

        for batch_no, batch in enumerate(iterator):
            logger.info("Processing batch n. %d", batch_no)
            coocc_dict = collections.defaultdict(int)
            catenae_freq = collections.defaultdict(int)

            for sentence_no, sentence in enumerate(batch):
                if sentence:
                    if not sentence_no % 100:
                        logger.info("%d - %d", sentence_no, len(sentence))
                    process_cooccurrences(sentence, coocc_dict,
                                          catenae_freq,
                                          accepted_catenae,
                                          min_len_catena, max_len_catena)

            filename_uuid = str(uuid.uuid4())

            sorted_cooc = sorted(coocc_dict.items())
            with open(output_dir.joinpath(f"catenae-coocc-{filename_uuid}"), "w") as fout_catenae:
                for cats, freq in sorted_cooc:
                    if freq > min_freq:
                        cat_i, cat_j = cats
                        print(f"{cat_i} {cat_j}\t{freq}", file=fout_catenae)

            sorted_cats = sorted(catenae_freq.items())
            total_freqs_partial = 0
            with open(output_dir.joinpath(f"catenae-freqs-{filename_uuid}"), "w") as fout_catenae:
                for cat, freq in sorted_cats:
                    total_freqs_partial += freq
                    print(f"{cat}\t{freq}", file=fout_catenae)

            total_freqs_global += total_freqs_partial

    fmerger.merge_and_collapse_iterable(futils.get_filenames_striterable(output_dir.glob("catenae-coocc-*")),
                                        output_filename=futils.get_str_path(output_dir / "catenae-coocc-summed.gz"),
                                        delete_input=True)
    fmerger.merge_and_collapse_iterable(futils.get_filenames_striterable(output_dir.glob("catenae-freqs-*")),
                                        output_filename=futils.get_str_path(output_dir / "catenae-freqs-summed.gz"),
                                        delete_input=True)

    with open(output_dir / "totals-freqs.txt", "wt") as fout_total:
        print(f"TOTAL\t{total_freqs_global}", file=fout_total)


def extract_catenae(output_dir: Path, input_dir: Path,
                    min_len_sentence: int, max_len_sentence: int, sentences_batch_size:int,
                    min_freq:int,
                    min_len_catena:int, max_len_catena:int) -> None:

    filenames = futils.get_filenames(input_dir)

    for filename in filenames:
        logger.info("Processing file %s", filename)
        iterator = dutils.grouper(cutils.plain_conll_reader(filename, min_len_sentence,
                                                            max_len_sentence),
                                  sentences_batch_size)

        for batch_no, batch in enumerate(iterator):
            logger.info("Processing batch n. %d", batch_no)
            freqdict = collections.defaultdict(int)
            catdict = collections.defaultdict(int)
            totalsdict = collections.defaultdict(int)

            for sentence_no, sentence in enumerate(batch):
                if sentence:
                    if not sentence_no % 100:
                        logger.info("%d - %d", sentence_no, len(sentence))
                    process_sentence(sentence, freqdict, catdict, totalsdict,
                                     min_len_catena, max_len_catena)

            filename_uuid = str(uuid.uuid4())
            with open(output_dir / f"catenae-freq-{filename_uuid}", "w") as fout_catenae, \
                    open(output_dir / f"items-freq-{filename_uuid}", "w") as fout_items, \
                    open(output_dir / f"totals-freq-{filename_uuid}", "w") as fout_totals:

                logger.info("Sorting catenae and printing...")

                sorted_catdict = sorted(catdict.items(), key=lambda x: x[0])
                sorted_freqdict = sorted(freqdict.items(), key=lambda x: x[0])
                sorted_totalsdict = sorted(totalsdict.items(), key=lambda x: str(x[0]))

                for catena, freq in sorted_catdict:
                    if freq > min_freq:
                        print(f"{catena}\t{freq}", file=fout_catenae)

                for item, freq in sorted_freqdict:
                    print(f"{item}\t{freq}", file=fout_items)

                for item, freq in sorted_totalsdict:
                    print(f"{item}\t{freq}", file=fout_totals)



    fmerger.merge_and_collapse_iterable(futils.get_filenames_striterable(output_dir.glob("catenae-freq-*")),
                                        output_filename=futils.get_str_path(output_dir / "catenae-freq-summed.gz"),
                                        delete_input=True)
    fmerger.merge_and_collapse_iterable(futils.get_filenames_striterable(output_dir.glob("items-freq-*")),
                                        output_filename=futils.get_str_path(output_dir / "items-freq-summed.gz"),
                                        delete_input=True)
    fmerger.merge_and_collapse_iterable(futils.get_filenames_striterable(output_dir.glob("totals-freq-*")),
                                        output_filename=futils.get_str_path(output_dir / "totals-freq-summed.gz"),
                                        delete_input=True)


def compute_mi(cur_line, freqdict_totals, freqdict_items):

    cur_linesplit = cur_line.strip().split("\t")
    cur_catena = cur_linesplit[0].split("|")
    cur_freq = float(cur_linesplit[1])

    p_catena = cur_freq / freqdict_totals[str(len(cur_catena))]
    p_els = 1
    for cur_item in cur_catena:
        p = freqdict_items[cur_item] / freqdict_totals['1'] # pylint: disable=invalid-name
        p_els *= p

    ret_mi = cur_freq * math.log(p_catena / p_els, 2)
    return ret_mi


def weight_catenae(output_dir, items_filepath, totals_filepath, catenae_filepath):

    freqdict_items = {}
    with gzip.open(items_filepath, "rt") as fin:
        logger.info("reading items...")
        for line in tqdm.tqdm(fin):
            linesplit = line.strip().split("\t")
            freqdict_items[linesplit[0]] = float(linesplit[1])

    freqdict_totals = {}
    with gzip.open(totals_filepath, "rt") as fin:
        logger.info("reading totals")
        for line in tqdm.tqdm(fin):
            linesplit = line.strip().split("\t")
            freqdict_totals[linesplit[0]] = float(linesplit[1])

    with gzip.open(catenae_filepath, "rt") as fin, \
         gzip.open(output_dir.joinpath("catenae-weighted.gz"), "wt") as fout:

        print("CATENA\tFREQ\tW", file=fout)
        logger.info("weighting catenae...")
        catenae_list = []
        for line in tqdm.tqdm(fin):
            linesplit = line.strip().split("\t")
            catena = linesplit[0].split("|")
            freq = float(linesplit[1])
            if len(catena) == 1:
                mutual_information = freq / freqdict_totals['1']
            else:
                mutual_information = compute_mi(line, freqdict_totals, freqdict_items)

            catenae_list.append((catena, freq, mutual_information))

        logger.info("sorting catenae based on mi...")
        sorted_catenae = sorted(catenae_list, key=lambda x: (-x[2], x[0]))

        for catena, freq, mutual_information in sorted_catenae:
            formatted_catena = "|".join(catena)
            print(f"{formatted_catena}\t{freq}\t{mutual_information}", file=fout)


def filter_catenae(output_dir, input_file, frequency_threshold, weight_threshold,
                   min_len_catena, max_len_catena):

    with gzip.open(input_file, "rt") as fin, \
        open(output_dir / "catenae-filtered.txt", "w") as fout, \
        open(output_dir / "catenae-lenone.txt", "w") as fout_words:

        print(fin.readline().strip(), file=fout)
        print(fin.readline().strip(), file=fout_words)

        for line in tqdm.tqdm(fin):
            line = line.strip().split("\t")
            catena, freq, weight = line
            catena = catena.split("|")
            freq = float(freq)
            weight = float(weight)

            if min_len_catena <= len(catena) <= max_len_catena and \
                freq > frequency_threshold and \
                weight > weight_threshold:

                formatted_catena = "|".join(catena)
                print(f"{formatted_catena}\t{freq}\t{weight}", file=fout)

            if len(catena) == 1:
                formatted_catena = "|".join(catena)
                print(f"{formatted_catena}\t{freq}\t{weight}", file=fout_words)


def extract_sentences(output_dir: Path, input_dir: Path, catenae_list_fname: Path) -> None:
    """_summary_

    Args:
        output_dir (Path): _description_
        input_dir (Path): _description_
        catenae_list_fname (Path): _description_
    """

    fout_list_sents = {}
    fout_list_cats = {}
    catenae_list = set()

    with open(catenae_list_fname) as f_catenae:
        for line in f_catenae:
            catena = line.strip()
            catenae_list.add(catena)
            fout_list_sents[catena] = open(output_dir.joinpath(f"{catena}.sentences"), "w")
            fout_list_cats[catena] = open(output_dir.joinpath(f"{catena}.cat"), "w")

    for input_file in tqdm.tqdm(input_dir.iterdir()):
        sentences_it = tqdm.tqdm(enumerate(cutils.plain_conll_reader(input_file.absolute(),
                                                                     min_len=1, max_len=25)))

        for _, sentence in sentences_it:

            if sentence:

                freqdict = collections.defaultdict(int)
                catdict = collections.defaultdict(int)
                totalsdict = collections.defaultdict(int)

                process_sentence(sentence, freqdict, catdict, totalsdict,
                                    min_len_catena=0, max_len_catena=5)

                catenae = catdict.keys()

                for catena in catenae_list:
                    if catena in catenae:
                        print("\n".join(sentence)+"\n", file=fout_list_sents[catena])
                        print("\n".join(catenae)+"\n", file=fout_list_cats[catena])
