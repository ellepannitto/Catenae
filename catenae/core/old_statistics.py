import os
import logging
import string
import collections

logger = logging.getLogger(__name__)


def compute_depth(startfrom, dep_dict):

    deplist = dep_dict[startfrom]

    if len(deplist):
        return 1 + max(compute_depth(x, dep_dict) for x in deplist)
    else:
        return 0


def reader(input_file):
    sentence = []
    with open(input_file) as fin:
        for line in fin:
            linestrip = line.strip()
            if len(linestrip):
                linesplit = linestrip.split("\t")
                sentence.append(linesplit)
            else:
                if len(sentence)>1:
                    yield sentence
                sentence = []
    if len(sentence)>1:
        yield(sentence)


def extract_stats(filename):
    admittedchars = string.ascii_letters + string.digits + " _-'!?"

    freqdict = collections.defaultdict(int)
    pos_freqdict = collections.defaultdict(int)
    verbmorph_freqdict = collections.defaultdict(lambda : collections.defaultdict(int))
    roots_freqdict = collections.defaultdict(int)
    n_questionmarks = 0
    n_sentences = 0
    n_tokens = 0
    n_chars = 0
    avg_depth = 0

    arity_of_verbal_roots = 0
    n_sentences += 1
    dependents = collections.defaultdict(list)
    verbal_root = False

    for sentence in reader(filename):
        for token in sentence:
            tok_id, form, lemma, _, PoS, morph, head, synrel, _, _ = token
            tok_id = int(tok_id)
            head = int(head)
            PoS = PoS[0]

            if all(c in admittedchars for c in lemma):
                n_tokens += 1
                n_chars += len(lemma)
                freqdict[lemma] += 1
                pos_freqdict[PoS] += 1

                n_questionmarks += 1

                if PoS == "V":
                    morph = morph.split("|")
                    for el in morph:
                        elsplit = el.split("=")
                        verbmorph_freqdict[elsplit[0]][elsplit[1]] += 1
                    if head == 0:
                        verbal_root = True
            else:
                # pass
                print(token)
                input()

            if head == 0:
                roots_freqdict[PoS] += 1

            dependents[head].append(tok_id)

        if verbal_root:
            arity_of_verbal_roots += len(dependents[dependents[0][0]])

        # depth = compute_depth(0, dependents)
        # avg_depth += depth

        # print(tok_id, form, lemma, PoS, morph, head, synrel)
        # input()


    VOC_SIZE = len(freqdict)
    AVG_SEN_LEN = n_tokens / n_sentences
    AVG_WORD_LEN = n_chars / n_tokens
    TTR = len(freqdict) / n_tokens
    HTR = len([x in freqdict for x in freqdict if freqdict[x] == 1]) / n_tokens
    LEXICAL_DENSITY = (pos_freqdict["N"] + pos_freqdict["V"] + pos_freqdict["J"] + pos_freqdict["R"]) / n_tokens
    AVG_DEPTH = avg_depth / n_sentences
    AVG_ARITY_OF_VERBAL_ROOTS = arity_of_verbal_roots / roots_freqdict["V"]

    print("SENTENCES NUMBER", n_sentences)
    print("WORD NUMBER", n_tokens)
    print("VOC SIZE: ", VOC_SIZE)
    print("AVG SENTENCE LENGTH", AVG_SEN_LEN)
    print("AVG WORD LENGTH", AVG_WORD_LEN)
    print("TTR", TTR)
    print("HTR", HTR)
    print("LEXICAL DENSITY", LEXICAL_DENSITY)
    print("POS DISTRIBUTION:")
    for key, value in pos_freqdict.items():
        print("  ", key, value / n_tokens)

    print(verbmorph_freqdict)

    print("AVG DEPTH", AVG_DEPTH)

    print("NUMBER OF VERBAL ROOTS OVER SENTENCES", roots_freqdict)
    print("AVG ARITY OF VERBAL ROOTS", AVG_ARITY_OF_VERBAL_ROOTS)


    # average depth of embedded complement chains, relative ordering of subordinate clauses, average length of
    # dependency links, mutual perplexity


def compute(output_dir, corpus_dirpath):

    for filename in os.listdir(corpus_dirpath):
        if filename.endswith(".conll"):
            extract_stats(corpus_dirpath+filename)