import logging
import string
import collections
import itertools
import uuid

from catenae.utils import files_utils as futils
from catenae.utils import data_utils as dutils
from catenae.utils import corpus_utils as cutils


logger = logging.getLogger(__name__)


def recursive_C(A, tree_children, th=5):
    # if A is a leaf
    if A not in tree_children:
        return [[A]], [[A]]

    else:
        found_catenae = []
        list_of_indep_catenae = [[[A]]]

        for a_child in tree_children[A]:
            c, all_c = recursive_C(a_child, tree_children)

            found_catenae += all_c
            list_of_indep_catenae.append([[None]] + c)

        X = []
        for tup in itertools.product(*list_of_indep_catenae):
            new_catena = list(sorted(filter(lambda x: x is not None, sum(tup, []))))
            if len(new_catena) <= th:
                X.append(new_catena)

        return X, X+found_catenae


def process(sentence, freqdict, catdict, totalsdict):
    admitted_chars = string.ascii_letters+".-' "

    children = {}
    tokens = {}
    postags = {}
    rels = {}
    tokens_to_remove = []

    for token in sentence:
        token = token.split("\t")

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
        _, catenae = recursive_C(root, children)

        for catena in catenae:
            if all(x not in tokens_to_remove for x in catena):
                print("CATENA:", catena)
                input()

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
                    print(cat)
                    if len(cat) > 1:
                        catdict[cat] += 1
                    totalsdict[(len(cat))] += 1


def extract_catenae(output_dir, input_dir,
                    min_len_sentence, max_len_sentence, sentences_batch_size,
                    min_freq):
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
                    process(sentence, freqdict, catdict, totalsdict)

            filename_uuid = str(uuid.uuid4())
            with open(out_dir + "/catenae-freq-" + filename_uuid, "w") as fout_catenae, \
                    open(out_dir + "/items-freq-" + filename_uuid, "w") as fout_items, \
                    open(out_dir + "/totals-freq-" + filename_uuid, "w") as fout_totals:

                print("sorting and printing")

                sorted_catdict = sorted(catdict.items(), key=lambda x: x[0])
                sorted_freqdict = sorted(freqdict.items(), key=lambda x: x[0])
                sorted_totalsdict = sorted(totalsdict.items(), key=lambda x: str(x[0]))

                for catena, freq in sorted_catdict:
                    if freq > min_freq:
                        print("{}\t{}".format(" ".join(catena), freq), file=fout_catenae)

                for item, freq in sorted_freqdict:
                    print("{}\t{}".format(item, freq), file=fout_items)

                for item, freq in sorted_totalsdict:
                    print("{}\t{}".format(item, freq), file=fout_totals)

    merge_and_collapse_pattern(out_dir+"/catenae-freq-*", output_filename=out_dir+"/catenae-freq-summed.txt")
    merge_and_collapse_pattern(out_dir+"/items-freq-*", output_filename=out_dir+"/items-freq-summed.txt")
    merge_and_collapse_pattern(out_dir+"/totals-freq-*", output_filename=out_dir+"/totals-freq-summed.txt")

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