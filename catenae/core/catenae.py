import os
import string
import itertools
import collections
import math
import uuid
import glob

# from old_project.recurrentbabbling.utils.FileMerger.filesmerger import merge_and_collapse_pattern, merge_and_collapse_iterable
from FileMerger.filesmerger import core as fmerger

from catenae.utils import corpus_utils as cutils
from catenae.utils import data_utils as dutils
from catenae.utils import catenae_utils as catutils


def process(sentence, freqdict, catdict, totalsdict):
    admitted_chars = string.ascii_letters+".-' "

    children = {}
    tokens = {}
    postags = {}
    rels = {}
    tokens_to_remove = []

    for token in sentence:

        position, word, lemma, pos, _, morph, head, rel, _, _ = token
        if not pos == "PUNCT" and rel not in ["discourse", "fixed", "flat", "comound", "list", "parataxis",
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
        _, catenae = catutils.recursive_C(root, children)

        for catena in catenae:
            if all(x not in tokens_to_remove for x in catena):
                # print("CATENA:", catena)
                # input()

                tokensandpostags = [[tokens[x] for x in catena],
                                    [postags[x] for x in catena],
                                    [rels[x] for x in catena]]
                # print(tokensandpostags)
                temp = [(0, 1, 2)] * len(catena)
                # temp = [[0]]*len(catena)
                X = list(itertools.product(*temp))

                for c in X:
                    # score = []
                    cat = []
                    for i, el in enumerate(c):
                        cat.append(tokensandpostags[el][i])
                    cat = tuple(cat)
                    if len(cat) > 1:
                        catdict[cat] += 1
                    totalsdict[(len(cat))] += 1


def extract_catenae(input_file, out_dir, sentences_batch_size, freq_threshold):

    iterator = dutils.grouper(cutils.plain_conll_reader(input_file, 1, 25), sentences_batch_size)

    for batch in iterator:

        print("NEW BATCH")
        freqdict = collections.defaultdict(int)
        catdict = collections.defaultdict(int)
        totalsdict = collections.defaultdict(int)
        for sentence_no, sentence in enumerate(batch):

            if sentence:
                if not sentence_no % 100:
                    print(sentence_no, len(sentence))
                process(sentence, freqdict, catdict, totalsdict)

        filename_uuid = str(uuid.uuid4())
        with open(out_dir+"/catenae-freq-"+filename_uuid, "w") as fout_catenae, \
                open(out_dir+"/items-freq-"+filename_uuid, "w") as fout_items, \
                open(out_dir+"/totals-freq-"+filename_uuid, "w") as fout_totals:

            print("sorting and printing")

            sorted_catdict = sorted(catdict.items(), key=lambda x: x[0])
            sorted_freqdict = sorted(freqdict.items(), key=lambda x: x[0])
            sorted_totalsdict = sorted(totalsdict.items(), key=lambda x: str(x[0]))

            for catena, freq in sorted_catdict:
                if freq > freq_threshold:
                    print("{}\t{}".format(" ".join(catena), freq), file=fout_catenae)

            for item, freq in sorted_freqdict:
                print("{}\t{}".format(item, freq), file=fout_items)

            for item, freq in sorted_totalsdict:
                print("{}\t{}".format(item, freq), file=fout_totals)

    fmerger.merge_and_collapse_pattern(out_dir+"/catenae-freq-*", output_filename=out_dir+"/catenae-freq-summed.txt")
    fmerger.merge_and_collapse_pattern(out_dir+"/items-freq-*", output_filename=out_dir+"/items-freq-summed.txt")
    fmerger.merge_and_collapse_pattern(out_dir+"/totals-freq-*", output_filename=out_dir+"/totals-freq-summed.txt")

    freqdict_items = {}
    with open(out_dir+"/items-freq-summed.txt") as fin:
        for line in fin:
            linesplit = line.strip().split("\t")
            freqdict_items[linesplit[0]] = float(linesplit[1])

    freqdict_totals = {}
    with open(out_dir+"/totals-freq-summed.txt") as fin:
        for line in fin:
            linesplit = line.strip().split("\t")
            freqdict_totals[linesplit[0]] = float(linesplit[1])

    def compute_mi(cur_line):

        cur_linesplit = cur_line.strip().split("\t")
        cur_catena = cur_linesplit[0].split(" ")
        cur_freq = float(cur_linesplit[1])

        p_catena = cur_freq / freqdict_totals[str(len(cur_catena))]
        p_els = 1
        for cur_item in cur_catena:
            p = freqdict_items[cur_item] / freqdict_totals['1']
            p_els *= p

        ret_mi = cur_freq * math.log(p_catena / p_els, 2)

        return ret_mi

    with open(out_dir + "/catenae-freq-summed.txt") as fin, open(out_dir+"/catenae-final.txt", "w") as fout:
        catenae_list = []
        for line in fin:
            linesplit = line.strip().split("\t")
            catena = linesplit[0].split(" ")
            freq = float(linesplit[1])
            mi = compute_mi(line)
            catenae_list.append((catena, freq, mi))

        sorted_catenae = sorted(catenae_list, key=lambda x: (-x[2], x[0]))

        for catena, freq, mi in sorted_catenae:
            print("{}\t{}\t{}".format(" ".join(catena), freq, mi), file=fout)


def split_lexical_and_categorical(output_folder, input_folder):

    files = glob.glob(input_folder+"/*.catenae")

    for file in files:
        prefix = file.split("/")[-1].split(".")[0]

        catenae = []
        catenae_lex = []
        catenae_syn = []
        catenae_mor = []
        catenae_misc = []

        with open(file) as fin:
            for line_no, line in enumerate(fin):
                linesplit = line.strip().split("\t")
                catena = linesplit[0]
                freq = float(linesplit[1])
                mi = float(linesplit[2])
                if mi > 0:
                    catenae.append((catena, freq, mi))

                    catena = catena.split(" ")
                    catena_firstel = [x[0] for x in catena]

                    if all(x in ["_"] for x in catena_firstel):
                        catenae_mor.append((" ".join(catena), freq, mi))
                    elif all(x in ["@"] for x in catena_firstel):
                        catenae_syn.append((" ".join(catena), freq, mi))
                    elif all(x not in ["_", "@"] for x in catena_firstel):
                        catenae_lex.append((" ".join(catena), freq, mi))
                    else:
                        catenae_misc.append((" ".join(catena), freq, mi))

        with open(output_folder+"/{}.lex".format(prefix), "w") as fout_lex, \
                open(output_folder+"/{}.mor".format(prefix), "w") as fout_mor, \
                open(output_folder+"/{}.syn".format(prefix), "w") as fout_syn, \
                open(output_folder+"/{}.misc".format(prefix), "w") as fout_misc,\
                open(output_folder+"/{}.sorted.freq".format(prefix), "w") as fout_sorted, \
                open(output_folder+"/{}.sorted.pmi".format(prefix), "w") as fout_sortedpmi:

            sorted_freq = sorted(catenae, key=lambda x: -x[1])
            for catena, freq, mi in sorted_freq:
                print("{}\t{}\t{}".format(catena, freq, mi), file=fout_sorted)
            for catena, freq, mi in catenae:
                print("{}\t{}\t{}".format(catena, freq, mi), file=fout_sortedpmi)
            for catena, freq, mi in catenae_lex:
                print("{}\t{}\t{}".format(catena, freq, mi), file=fout_lex)
            for catena, freq, mi in catenae_mor:
                print("{}\t{}\t{}".format(catena, freq, mi), file=fout_mor)
            for catena, freq, mi in catenae_syn:
                print("{}\t{}\t{}".format(catena, freq, mi), file=fout_syn)
            for catena, freq, mi in catenae_misc:
                print("{}\t{}\t{}".format(catena, freq, mi), file=fout_misc)


def process_abstraction(sentence, cooccurrences, accepted_catenae):

    admitted_chars = string.ascii_letters + ".-' "

    children = {}
    tokens = {}
    postags = {}
    rels = {}
    tokens_to_remove = []

    for token in sentence:

        position, word, lemma, pos, _, morph, head, rel, _, _ = token
        if not pos == "PUNCT" and rel not in ["discourse", "fixed", "flat", "comound", "list", "parataxis",
                                              "orphan", "goeswith", "reparandum", "punct", "dep", "mark"]:
            position = int(position)

            if not all(c in admitted_chars for c in word):
                tokens_to_remove.append(position)

            head = int(head)
            if head not in children:
                children[head] = []

            children[head].append(position)
            tokens[position] = word
            postags[position] = "_" + pos
            rels[position] = "@" + rel

    if 0 in children:
        root = children[0][0]
        _, catenae = dutils.recursive_C(root, children, 3)

        for catena in catenae:

            explicit_catenae = {}
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
                    if len(cat) > 1:
                        cat = "|".join(cat)
                        if cat in accepted_catenae:
                            explicit_catenae[c] = cat

            for el1 in explicit_catenae:
                for el2 in explicit_catenae:
                    if not el1 == el2:
                        if all(x <= y for x, y in zip(el1, el2)):
                            tup = (explicit_catenae[el1], explicit_catenae[el2])
                            cooccurrences[tup] += 1


def parallel_abstraction_chain(output_folder, accepted_catenae, batch):

    print(os.getpid(), "NEW BATCH")

    filename_uuid = str(uuid.uuid4())
    cooccurrences = collections.defaultdict(int)

    for sentence_no, sentence in enumerate(batch):
        if sentence:
            if not sentence_no % 1000:
                print(os.getpid(), sentence_no, len(sentence))
            process_abstraction(sentence, cooccurrences, accepted_catenae)

    sorted_cooc = sorted(cooccurrences.items())
    with open(output_folder + "/catenae-coocc-" + filename_uuid, "w") as fout_catenae:
        for cats, freq in sorted_cooc:
            cat_i, cat_j = cats
            print("{} {}\t{}".format(cat_i, cat_j, freq), file=fout_catenae)

    return output_folder + "/catenae-coocc-" + filename_uuid


def abstraction_chains(output_folder, input_file, accepted_catenae, sentences_batch_size):

    iterator = dutils.grouper(cutils.plain_conll_reader(input_file, 1, 25), sentences_batch_size)

    files_to_merge = []
    merge_steps = 0

    for batch in iterator:
        new_file = parallel_abstraction_chain(output_folder, accepted_catenae, batch)
        files_to_merge.append(new_file)

        if len(files_to_merge) > 30:
            new_filename = output_folder+"/merged-{}".format(merge_steps)
            fmerger.merge_and_collapse_iterable(files_to_merge, output_folder+"/merged-{}".format(merge_steps))

            merge_steps += 1
            files_to_merge = [new_filename]

    fmerger.merge_and_collapse_iterable(files_to_merge, output_folder + "/merged-{}".format(merge_steps))
