import os
import logging
import string
import collections

logger = logging.getLogger(__name__)

def reader(input_file):
    sentence = []
    with open(input_file) as fin:
        for line in fin:
            linestrip = line.strip()
            if len(linestrip) and not linestrip[0]=="#":
                linesplit = linestrip.split("\t")
                sentence.append(linesplit)
            else:
                if len(sentence)>1:
                    yield sentence
                sentence = []
    if len(sentence)>1:
        yield(sentence)


def extract_pos(corpus_filepath, pos):

    tot = 0
    freqdist = collections.defaultdict(int)
    for sentence in reader(corpus_filepath):
        for token in sentence:
            token_pos = token[3]
            if token_pos == pos:
                # print(pos)
                freqdist[token[2]] += 1
                tot+=1


    sorted_freqdist = sorted(freqdist.items(), key = lambda x: -x[1])
    sorted_freqdist = [(x, y, "{:.10f}%".format(y*100/tot)) for x, y in sorted_freqdist]
    return sorted_freqdist


def extract_tense(corpus_filepath, tense):

    tot = 0
    freqdist = collections.defaultdict(int)
    for sentence in reader(corpus_filepath):
        for token in sentence:
            token_pos = token[3]
            if token_pos == "VERB":
                token_morph = token[5].split("|")
                if len(token_morph)>0:
                    for trait in token_morph:
                        name, value = trait.split("=")
                        if name == "Tense" and value == tense:
                            freqdist[token[2]] += 1
                            tot+=1
    sorted_freqdist = sorted(freqdist.items(), key = lambda x: -x[1])
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
    sorted_freqdist = sorted(freqdist.items(), key = lambda x: -x[1])
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
    sorted_freqdist = sorted(freqdist.items(), key = lambda x: -x[1])
    sorted_freqdist = [(x, y, "{:.10f}%".format(y*100/tot)) for x, y in sorted_freqdist]
    return sorted_freqdist




def compute_pos_distribution(output_dir, corpus_dirpath, pos_list):

    for pos in pos_list:
        for filename in os.listdir(corpus_dirpath):
            if filename.endswith(".conll"):
                logger.info("Analyzing corpus: {}".format(filename))
                sorted_freqs = extract_pos(corpus_dirpath+filename, pos)

                with open(output_dir+corpus_dirpath.split("/")[-2]+"."+filename.split(".")[0]+".{}".format(pos), "w") as fout:
                    for x, y, z in sorted_freqs:
                        print("{}\t{}\t{}".format(x, y, z), file=fout)


def compute_tense_distribution(output_dir, corpus_dirpath, tense_list):

    for tense in tense_list:
        for filename in os.listdir(corpus_dirpath):
            if filename.endswith(".conll"):
                logger.info("Analyzing corpus: {}".format(filename))
                sorted_freqs = extract_tense(corpus_dirpath+filename, tense)

                with open(output_dir+corpus_dirpath.split("/")[-2]+"."+filename.split(".")[0]+".{}".format(tense), "w") as fout:
                    for x, y, z in sorted_freqs:
                        print("{}\t{}\t{}".format(x, y, z), file=fout)


def compute_mood_distribution(output_dir, corpus_dirpath, mood_list):

    for mood in mood_list:
        for filename in os.listdir(corpus_dirpath):
            if filename.endswith(".conll"):
                logger.info("Analyzing corpus: {}".format(filename))
                sorted_freqs = extract_mood(corpus_dirpath+filename, mood)

                with open(output_dir+corpus_dirpath.split("/")[-2]+"."+filename.split(".")[0]+".{}".format(mood), "w") as fout:
                    for x, y, z in sorted_freqs:
                        print("{}\t{}\t{}".format(x, y, z), file=fout)


def compute_form_distribution(output_dir, corpus_dirpath, form_list):

    for form in form_list:
        for filename in os.listdir(corpus_dirpath):
            if filename.endswith(".conll"):
                logger.info("Analyzing corpus: {}".format(filename))
                sorted_freqs = extract_form(corpus_dirpath+filename, form)

                with open(output_dir+corpus_dirpath.split("/")[-2]+"."+filename.split(".")[0]+".{}".format(form), "w") as fout:
                    for x, y, z in sorted_freqs:
                        print("{}\t{}\t{}".format(x, y, z), file=fout)