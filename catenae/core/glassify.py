import logging
import tqdm
import glob
import string
import itertools
import math
import collections

from catenae.utils import corpus_utils as cutils
from catenae.utils import data_utils as dutils
from catenae.core import extraction


logger = logging.getLogger(__name__)


def filter(output_dir: str, input_dir: str, catenae_fpath: str,
           min_len_catena: int, max_len_catena: int) -> None:

    catenae_glass = dutils.load_catenae_set(catenae_fpath, math.inf)

    admitted_chars = string.ascii_letters+".-' "

    input_files = glob.glob(input_dir+"/*")
    input_files_it = tqdm.tqdm(input_files)
    
    for input_file in input_files_it:
        input_files_it.set_description(f"Reading file {input_file}")

        sentences_it = tqdm.tqdm(enumerate(cutils.PlainCoNLLReader(input_file, min_len=1, max_len=25)))

        for _, sentence in sentences_it:
            print("\n".join(str(x) for x in sentence))
            
            if sentence:
                children = {}
                tokens = {}
                postags = {}
                rels = {}
                tokens_to_remove = []

                CATENAE = collections.defaultdict(list)

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

                if 0 in children:
                    root = children[0][0]
                    _, catenae = extraction.recursive_C(root, children, min_len_catena, max_len_catena)

                    explicit_catenae = set()

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
                                explicit_catenae.add(cat)
                        
                        print(catena)
                        print("|".join(cat))

                        cat_to_look_for = "|".join(cat)
                        
                        if cat_to_look_for in catenae_glass:
                            CATENAE[tuple(catena)].append(cat)
                            print(CATENAE)
                        else:
                            print("catena not present")
                        input()                            


