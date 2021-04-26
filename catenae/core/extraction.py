import logging
import string
import collections
import itertools
import uuid
import glob
import gzip
import math

from catenae.utils import files_utils as futils
from catenae.utils import data_utils as dutils
from catenae.utils import corpus_utils as cutils

from FileMerger.filesmerger import core as fmerger


logger = logging.getLogger(__name__)


def recursive_C(A, tree_children, min_len_catena, max_len_catena):
    # if A is a leaf
    if A not in tree_children:
        return [[A]], [[A]]
        # return [[A]], []

    else:
        found_catenae = []
        list_of_indep_catenae = [[[A]]]

        for a_child in tree_children[A]:
            c, all_c = recursive_C(a_child, tree_children, min_len_catena, max_len_catena)

            found_catenae += all_c
            list_of_indep_catenae.append([[None]] + c)

        X = []
        for tup in itertools.product(*list_of_indep_catenae):
            new_catena = list(sorted(filter(lambda x: x is not None, sum(tup, []))))
            if min_len_catena <= len(new_catena) <= max_len_catena:
                X.append(new_catena)

        return X, X+found_catenae


def process_sentence(sentence, freqdict, catdict, totalsdict,
                     min_len_catena, max_len_catena):
    admitted_chars = string.ascii_letters+".-' "

    children = {}
    tokens = {}
    postags = {}
    rels = {}
    tokens_to_remove = []

    for token in sentence:
        token = token.split("\t")

        position, word, lemma, pos, _, morph, head, rel, _, _ = token
        if not pos == 'PUNCT' and rel not in ["discourse", "fixed", "flat", "comound", "list", "parataxis",
                                              "orphan", "goeswith", "reparandum", "punct", "dep"]:
            position = int(position)

            if not all(c in admitted_chars for c in word):
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
        _, catenae = recursive_C(root, children, min_len_catena, max_len_catena)

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

                    # if len(cat) > 1:
                    catdict["|".join(cat)] += 1
                    totalsdict[(len(cat))] += 1


def process_cooccurrences(sentence, coocc_dict, catenae_freq,
                          accepted_catenae, min_len_catena, max_len_catena):

    admitted_chars = string.ascii_letters+".-' "

    children = {}
    tokens = {}
    postags = {}
    rels = {}
    tokens_to_remove = []

    for token in sentence:
        token = token.split("\t")
        position, word, lemma, pos, _, morph, head, rel, _, _ = token
        if not pos == "PUNCT" and not rel in ["discourse", "fixed", "flat", "comound", "list", "parataxis",
                                              "orphan", "goeswith", "reparandum", "punct", "dep", "mark"]:
            position = int(position)

            if not all(c in admitted_chars for c in word):
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
        _, catenae = recursive_C(root, children, min_len_catena, max_len_catena)

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


def extract_coccurrences(output_dir, input_dir, accepted_catenae_filepath, top_k,
                         min_len_sentence, max_len_sentence, sentences_batch_size,
                         min_freq, min_len_catena, max_len_catena,
                         include_words, words_filepath):

    # TODO: add parameter for top K
    accepted_catenae = dutils.load_catenae_set(accepted_catenae_filepath, top_k)

    if include_words:
        accepted_catenae = dutils.load_catenae_set(words_filepath, float('inf'), accepted_catenae)

    filenames = futils.get_filenames(input_dir)
    total_freqs_global = 0

    for filename in filenames:
        logger.info("Processing file {}".format(filename))
        iterator = dutils.grouper(cutils.PlainCoNLLReader(filename, min_len_sentence, max_len_sentence),
                                  sentences_batch_size)

        for batch_no, batch in enumerate(iterator):
            logger.info("Processing batch n. {}".format(batch_no))
            coocc_dict = collections.defaultdict(int)
            catenae_freq = collections.defaultdict(int)

            for sentence_no, sentence in enumerate(batch):
                if sentence:
                    if not sentence_no % 100:
                        logger.info("{} - {}".format(sentence_no, len(sentence)))
                    process_cooccurrences(sentence, coocc_dict,
                                          catenae_freq,
                                          accepted_catenae,
                                          min_len_catena, max_len_catena)

            filename_uuid = str(uuid.uuid4())

            sorted_cooc = sorted(coocc_dict.items())
            with open(output_dir + "/catenae-coocc-" + filename_uuid, "w") as fout_catenae:
                for cats, freq in sorted_cooc:
                    if freq > min_freq:
                        cat_i, cat_j = cats
                        print("{} {}\t{}".format(cat_i, cat_j, freq), file=fout_catenae)

            sorted_cats = sorted(catenae_freq.items())
            total_freqs_partial = 0
            with open(output_dir + "/catenae-freqs-" + filename_uuid, "w") as fout_catenae:
                for cat, freq in sorted_cats:
                    total_freqs_partial += freq
                    print("{}\t{}".format(cat, freq), file=fout_catenae)

            total_freqs_global += total_freqs_partial

    fmerger.merge_and_collapse_iterable(glob.iglob(output_dir + "/catenae-coocc-*"),
                                        output_filename=output_dir + "/catenae-coocc-summed.gz",
                                        delete_input=True)
    fmerger.merge_and_collapse_iterable(glob.iglob(output_dir + "/catenae-freqs-*"),
                                        output_filename=output_dir + "/catenae-freqs-summed.gz",
                                        delete_input=True)

    with open(output_dir + "/totals-freqs.txt", "wt") as fout_total:
        print("TOTAL\t{}".format(total_freqs_global), file=fout_total)


def extract_catenae(output_dir, input_dir,
                    min_len_sentence, max_len_sentence, sentences_batch_size,
                    min_freq,
                    min_len_catena, max_len_catena):

    filenames = futils.get_filenames(input_dir)

    for filename in filenames:
        logger.info("Processing file {}".format(filename))
        iterator = dutils.grouper(cutils.PlainCoNLLReader(filename, min_len_sentence, max_len_sentence),
                                  sentences_batch_size)

        for batch_no, batch in enumerate(iterator):
            logger.info("Processing batch n. {}".format(batch_no))
            freqdict = collections.defaultdict(int)
            catdict = collections.defaultdict(int)
            totalsdict = collections.defaultdict(int)

            for sentence_no, sentence in enumerate(batch):
                if sentence:
                    if not sentence_no % 100:
                        logger.info("{} - {}".format(sentence_no, len(sentence)))
                    process_sentence(sentence, freqdict, catdict, totalsdict,
                                     min_len_catena, max_len_catena)

            filename_uuid = str(uuid.uuid4())
            with open(output_dir + "/catenae-freq-" + filename_uuid, "w") as fout_catenae, \
                    open(output_dir + "/items-freq-" + filename_uuid, "w") as fout_items, \
                    open(output_dir + "/totals-freq-" + filename_uuid, "w") as fout_totals:

                logger.info("Sorting catenae and printing")

                sorted_catdict = sorted(catdict.items(), key=lambda x: x[0])
                sorted_freqdict = sorted(freqdict.items(), key=lambda x: x[0])
                sorted_totalsdict = sorted(totalsdict.items(), key=lambda x: str(x[0]))

                for catena, freq in sorted_catdict:
                    if freq > min_freq:
                        print("{}\t{}".format(catena, freq), file=fout_catenae)

                for item, freq in sorted_freqdict:
                    print("{}\t{}".format(item, freq), file=fout_items)

                for item, freq in sorted_totalsdict:
                    print("{}\t{}".format(item, freq), file=fout_totals)

    fmerger.merge_and_collapse_iterable(glob.iglob(output_dir + "/catenae-freq-*"),
                                        output_filename=output_dir + "/catenae-freq-summed.gz",
                                        delete_input=True)
    fmerger.merge_and_collapse_iterable(glob.iglob(output_dir + "/items-freq-*"),
                                        output_filename=output_dir + "/items-freq-summed.gz",
                                        delete_input=True)
    fmerger.merge_and_collapse_iterable(glob.iglob(output_dir + "/totals-freq-*"),
                                        output_filename=output_dir + "/totals-freq-summed.gz",
                                        delete_input=True)


def compute_mi(cur_line, freqdict_totals, freqdict_items):

    cur_linesplit = cur_line.strip().split("\t")
    cur_catena = cur_linesplit[0].split("|")
    cur_freq = float(cur_linesplit[1])

    p_catena = cur_freq / freqdict_totals[str(len(cur_catena))]
    p_els = 1
    for cur_item in cur_catena:
        p = freqdict_items[cur_item] / freqdict_totals['1']
        p_els *= p

    ret_mi = cur_freq * math.log(p_catena / p_els, 2)
    return ret_mi


def weight_catenae(output_dir, items_filepath, totals_filepath, catenae_filepath):

    freqdict_items = {}
    with gzip.open(items_filepath, "rt") as fin:
        for line in fin:
            linesplit = line.strip().split("\t")
            freqdict_items[linesplit[0]] = float(linesplit[1])

    freqdict_totals = {}
    with gzip.open(totals_filepath, "rt") as fin:
        for line in fin:
            linesplit = line.strip().split("\t")
            freqdict_totals[linesplit[0]] = float(linesplit[1])

    with gzip.open(catenae_filepath, "rt") as fin, open(output_dir + "/catenae-weighted.txt", "w") as fout:
        catenae_list = []
        for line in fin:
            linesplit = line.strip().split("\t")
            catena = linesplit[0].split("|")
            freq = float(linesplit[1])
            if len(catena) == 1:
                mi = freq / freqdict_totals['1']
            else:
                mi = compute_mi(line, freqdict_totals, freqdict_items)
            catenae_list.append((catena, freq, mi))

        sorted_catenae = sorted(catenae_list, key=lambda x: (-x[2], x[0]))

        print("CATENA\tFREQ\tW", file=fout)
        for catena, freq, mi in sorted_catenae:
            print("{}\t{}\t{}".format("|".join(catena), freq, mi), file=fout)


def filter_catenae(output_dir, input_file, frequency_threshold, weight_threshold,
                              min_len_catena, max_len_catena):

    with open(input_file) as fin, open(output_dir+"/catenae-filtered.txt", "w") as fout:
        print(fin.readline().strip(), file=fout)
        for line in fin:
            line = line.strip().split("\t")
            catena, freq, weight = line
            catena = catena.split("|")
            freq = float(freq)
            weight = float(weight)

            if min_len_catena <= len(catena) <= max_len_catena and \
                freq > frequency_threshold and \
                weight > weight_threshold:

                print("{}\t{}\t{}".format("|".join(catena), freq, weight), file=fout)
