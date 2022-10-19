import os
import logging
import string
import collections
import itertools

logger = logging.getLogger(__name__)


def reader(input_file):
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


def extract_rel(corpus_filepath, synrel):
    tot_lemmatized = 0
    freqdist_lemmatized = collections.defaultdict(int)

    tot_mixed = 0
    freqdist_mixed = collections.defaultdict(int)

    tot_abstract = 0
    freqdist_abstract = collections.defaultdict(int)

    for sentence in reader(corpus_filepath):
        tree = collections.defaultdict(int)
        items = []

        for token in sentence:
            id, _, lemma, pos, _, _, head, rel, _, _ = token
            id = int(id)
            head = int(head)
            tree[id - 1] = head - 1

            if rel == synrel:
                items.append(id-1)

        # print("\n".join([" ".join(tok) for tok in sentence]))
        # print(items)

        for it in items:
            minpos, maxpos = min(it, tree[it]), max(it, tree[it])

            pair_lemma = (sentence[minpos][2], sentence[maxpos][2])
            pair_pos = ("@"+sentence[minpos][3], "@"+sentence[maxpos][3])
            pair_rel = ("_"+sentence[minpos][7], "_"+sentence[maxpos][7])

            pairs_list = [pair_lemma, pair_pos, pair_rel]
            combinations = list(itertools.combinations_with_replacement([0, 1, 2], 2))

            # print(minpos, maxpos)
            # print(pair_lemma, pair_pos, pair_rel)

            combinations_list = []
            for a, b in combinations:
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
    sorted_freqdist_lemmatized = [(x, y, "{:.10f}%".format(y * 100 / tot_lemmatized))
                                  for x, y in sorted_freqdist_lemmatized]

    sorted_freqdist_mixed = sorted(freqdist_mixed.items(), key=lambda x: -x[1])
    sorted_freqdist_mixed = [(x, y, "{:.10f}%".format(y * 100 / tot_mixed)) for x, y in sorted_freqdist_mixed]

    sorted_freqdist_abstract = sorted(freqdist_abstract.items(), key=lambda x: -x[1])
    sorted_freqdist_abstract = [(x, y, "{:.10f}%".format(y * 100 / tot_abstract)) for x, y in sorted_freqdist_abstract]

    return sorted_freqdist_lemmatized, sorted_freqdist_mixed, sorted_freqdist_abstract


def extract_edges(corpus_filepath, n):
    tot = 0
    freqdist = collections.defaultdict(int)

    for sentence in reader(corpus_filepath):
        tree = collections.defaultdict(list)
        verbs = []
        for token in sentence:
            id, _, lemma, pos, _, _, head, rel, _, _ = token
            id = int(id)
            head = int(head)
            if pos == "VERB":
                verbs.append(id-1)
            if not pos == "PUNCT":
                tree[head - 1].append(id - 1)

        for verb in verbs:
            if len(tree[verb]) == n:
                # print(sentence[verb])

                freqdist[sentence[verb][2]] += 1
                tot += 1

    sorted_freqdist = sorted(freqdist.items(), key=lambda x: -x[1])
    sorted_freqdist = [(x, y, "{:.10f}%".format(y * 100 / tot)) for x, y in sorted_freqdist]
    return sorted_freqdist


def extract_subj_verb(corpus_filepath):
    tot_pre = 0
    freqdist_pre = collections.defaultdict(int)
    tot_post = 0
    freqdist_post = collections.defaultdict(int)

    for sentence in reader(corpus_filepath):
        tree = collections.defaultdict(int)
        sbjs = []
        for token in sentence:
            id, _, lemma, pos, _, _, head, rel, _, _ = token
            id = int(id)
            head = int(head)
            if rel == "nsubj":
                sbjs.append(id - 1)
                tree[id-1] = head - 1

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
    sorted_freqdist_pre = [(x, y, "{:.10f}%".format(y * 100 / tot_pre)) for x, y in sorted_freqdist_pre]

    sorted_freqdist_post = sorted(freqdist_post.items(), key=lambda x: -x[1])
    sorted_freqdist_post = [(x, y, "{:.10f}%".format(y * 100 / tot_post)) for x, y in sorted_freqdist_post]

    return sorted_freqdist_pre, sorted_freqdist_post


def extract_subj_noun(corpus_filepath):
    tot_pre = 0
    freqdist_pre = collections.defaultdict(int)
    tot_post = 0
    freqdist_post = collections.defaultdict(int)

    for sentence in reader(corpus_filepath):
        tree = collections.defaultdict(int)
        sbjs = []
        for token in sentence:
            id, _, lemma, pos, _, _, head, rel, _, _ = token
            id = int(id)
            head = int(head)
            if rel == "nsubj":
                sbjs.append(id - 1)
                tree[id-1] = head - 1

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
    sorted_freqdist_pre = [(x, y, "{:.10f}%".format(y * 100 / tot_pre)) for x, y in sorted_freqdist_pre]

    sorted_freqdist_post = sorted(freqdist_post.items(), key=lambda x: -x[1])
    sorted_freqdist_post = [(x, y, "{:.10f}%".format(y * 100 / tot_post)) for x, y in sorted_freqdist_post]

    return sorted_freqdist_pre, sorted_freqdist_post


def extract_pos(corpus_filepath, pos):

    tot = 0
    freqdist = collections.defaultdict(int)
    for sentence in reader(corpus_filepath):
        for token in sentence:
            token_pos = token[3]
            if token_pos == pos:
                # print(pos)
                freqdist[token[2]] += 1
                tot += 1

    sorted_freqdist = sorted(freqdist.items(), key=lambda x: -x[1])
    sorted_freqdist = [(x, y, "{:.10f}%".format(y*100/tot)) for x, y in sorted_freqdist]
    return sorted_freqdist


def extract_morph(corpus_filepath, morph_trait, morph_value):

    tot = 0
    freqdist = collections.defaultdict(int)
    for sentence in reader(corpus_filepath):
        for token in sentence:
            token_pos = token[3]
            if token_pos == "VERB":
                token_morph = token[5].split("|")
                if len(token_morph) > 0:
                    for trait in token_morph:
                        name, value = trait.split("=")
                        if name == morph_trait and value == morph_value:
                            freqdist[token[2]] += 1
                            tot += 1
    sorted_freqdist = sorted(freqdist.items(), key=lambda x: -x[1])
    sorted_freqdist = [(x, y, "{:.10f}%".format(y*100/tot)) for x, y in sorted_freqdist]
    return sorted_freqdist


def extract_mood(corpus_filepath, mood):

    tot = 0
    freqdist = collections.defaultdict(int)
    for sentence in reader(corpus_filepath):
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
    sorted_freqdist = [(x, y, "{:.10f}%".format(y*100/tot)) for x, y in sorted_freqdist]
    return sorted_freqdist


def extract_form(corpus_filepath, form):

    tot = 0
    freqdist = collections.defaultdict(int)
    for sentence in reader(corpus_filepath):
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
    sorted_freqdist = [(x, y, "{:.10f}%".format(y*100/tot)) for x, y in sorted_freqdist]
    return sorted_freqdist


def compute_pos_distribution(output_dir, corpus_dirpath, pos_list):

    for pos in pos_list:
        for filename in os.listdir(corpus_dirpath):
            if filename.endswith(".conll"):
                logger.info("Analyzing corpus: {}".format(filename))
                sorted_freqs = extract_pos(corpus_dirpath+filename, pos)

                with open(output_dir.joinpath(corpus_dirpath.split("/")[-2]+"."+filename.split(".")[0]+f".{pos}"),
                          "w") as fout:
                    for x, y, z in sorted_freqs:
                        print("{}\t{}\t{}".format(x, y, z), file=fout)


def compute_morph_distribution(output_dir, corpus_dirpath, trait, values_list):

    for value in values_list:
        for filename in os.listdir(corpus_dirpath):
            if filename.endswith(".conll"):
                logger.info("Analyzing corpus: {}".format(filename))
                sorted_freqs = extract_morph(corpus_dirpath+filename, trait, value)

                with open(output_dir + corpus_dirpath.split("/")[-2] + "." + filename.split(".")[0] +
                          ".{}_{}".format(trait, value), "w") as fout:
                    for x, y, z in sorted_freqs:
                        print("{}\t{}\t{}".format(x, y, z), file=fout)


def compute_verbedges_distribution(output_dir, corpus_dirpath, number_edges):
    for filename in os.listdir(corpus_dirpath):
        if filename.endswith(".conll"):
            logger.info("Analyzing corpus: {}".format(filename))
            sorted_freqs = extract_edges(corpus_dirpath + filename, number_edges)

            with open(output_dir + corpus_dirpath.split("/")[-2] + "." + filename.split(".")[0] +
                      ".{}edge".format(number_edges), "w") as fout:
                for x, y, z in sorted_freqs:
                    print("{}\t{}\t{}".format(x, y, z), file=fout)


def compute_sbj_distribution(output_dir, corpus_dirpath):

    for filename in os.listdir(corpus_dirpath):
        if filename.endswith(".conll"):
            logger.info("Analyzing corpus: {}".format(filename))
            sorted_freqs_pre, sorted_freqs_post = extract_subj_verb(corpus_dirpath + filename)

            with open(output_dir + corpus_dirpath.split("/")[-2] + "." + filename.split(".")[0] + ".presubjverb",
                      "w") as fout_pre, \
                open(output_dir + corpus_dirpath.split("/")[-2] + "." + filename.split(".")[0] + ".postsubjverb",
                     "w") as fout_post:
                for x, y, z in sorted_freqs_pre:
                    print("{}\t{}\t{}".format(x, y, z), file=fout_pre)
                for x, y, z in sorted_freqs_post:
                    print("{}\t{}\t{}".format(x, y, z), file=fout_post)

            sorted_freqs_pre, sorted_freqs_post = extract_subj_noun(corpus_dirpath + filename)

            with open(output_dir + corpus_dirpath.split("/")[-2] + "." + filename.split(".")[0] + ".presubjnoun",
                      "w") as fout_pre, \
                open(output_dir + corpus_dirpath.split("/")[-2] + "." + filename.split(".")[0] + ".postsubjnoun",
                     "w") as fout_post:
                for x, y, z in sorted_freqs_pre:
                    print("{}\t{}\t{}".format(x, y, z), file=fout_pre)
                for x, y, z in sorted_freqs_post:
                    print("{}\t{}\t{}".format(x, y, z), file=fout_post)


def compute_synrel_distribution(output_dir, corpus_dirpath, synrel_list):

    for rel in synrel_list:
        for filename in os.listdir(corpus_dirpath):
            if filename.endswith(".conll"):
                logger.info("Analyzing corpus: {}".format(filename))
                sorted_freqs_lem, sorted_freqs_mix, sorted_freqs_abs = extract_rel(corpus_dirpath+filename, rel)

                with open(output_dir.joinpath(corpus_dirpath.split("/")[-2]+"."+filename.split(".")[0]+f".lem_{rel}"),
                          "w") as fout_lem, \
                     open(output_dir + corpus_dirpath.split("/")[-2] + "." + filename.split(".")[
                            0] + ".mix_{}".format(rel), "w") as fout_mix, \
                     open(output_dir + corpus_dirpath.split("/")[-2] + "." + filename.split(".")[
                            0] + ".abs_{}".format(rel), "w") as fout_abs:

                    for x, y, z in sorted_freqs_lem:
                        print("{}\t{}\t{}".format(x, y, z), file=fout_lem)
                    for x, y, z in sorted_freqs_mix:
                        print("{}\t{}\t{}".format(x, y, z), file=fout_mix)
                    for x, y, z in sorted_freqs_abs:
                        print("{}\t{}\t{}".format(x, y, z), file=fout_abs)
