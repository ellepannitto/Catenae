# pylint: disable=unspecified-encoding
import logging
import itertools
import math
import functools
import collections
import time

from pathlib import Path
from multiprocessing import Pool

from typing import Iterable, Tuple, List, Dict

import tqdm
from tqdm.contrib.concurrent import process_map
import networkx as nx
import networkit as nit

from catenae.utils import corpus_utils as cutils
from catenae.utils import data_utils as dutils
from catenae.utils import catenae_utils as catutils
from catenae.utils import glassify_utils as gutils


logger = logging.getLogger(__name__)


def compute_matrix(output_dir: Path, input_dir: Path, catenae_fpath: Path,
                   min_len_catena: int, max_len_catena: int,
                   multiprocess: bool, n_workers: int) -> None:
    """_summary_

    Args:
        output_dir (Path): _description_
        input_dir (Path): _description_
        catenae_fpath (Path): _description_
        min_len_catena (int): _description_
        max_len_catena (int): _description_
        multiprocess (bool): _description_
        n_workers (int): _description_
    """

    # TODO implement multiprocess

    catenae_glass = dutils.load_catenae_set(catenae_fpath, math.inf)

    input_files_it = tqdm.tqdm(input_dir.iterdir())

    for input_file in input_files_it:

        with open(output_dir.joinpath(input_file.name), "w") as fout:
            input_files_it.set_description(f"Reading file {input_file}")

            for _, sentence in enumerate(cutils.plain_conll_reader(input_file.absolute(),
                                                                   min_len=1, max_len=25)):
                if sentence:
                    children = {}
                    tokens = {}
                    postags = {}
                    rels = {}
                    tokens_to_remove = []

                    projected_sentence = [dutils.DefaultList([el.split("\t")[1]], "_")
                                          for el in sentence]
                    currently_looking_at = 1

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
                                                                           min_len_catena,
                                                                           max_len_catena)

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

                                    cat_to_look_for = "|".join(cat)

                                    if cat_to_look_for in catenae_glass:

                                        for i, label in zip(catena, cat):
                                            projected_sentence[i-1][currently_looking_at] = label
                                        currently_looking_at += 1

                        for lst in projected_sentence:
                            lst.fill(currently_looking_at)

                        print("\n".join("\t".join(x) for x in projected_sentence), file=fout)
                        print("\n", file=fout)


def process_old(search_space, len_sentence): #NETWORKX

    solutions = collections.defaultdict(list)

    G = nx.Graph()
    for i, _ in search_space.items():
        G.add_node(i)

    number_of_connections = 0
    for i, cxn1 in search_space.items():
        for j, cxn2 in search_space.items():
            if j>i:
                if gutils.matchable(cxn1, cxn2):
                    number_of_connections += 1
                    G.add_edge(i, j)

    for element in nx.find_cliques(G):
        projected_sentence = ["_"]*len_sentence

        for idx in element:
            projected_sentence, _ = gutils.update(projected_sentence, search_space[idx])

        solutions[tuple(projected_sentence)].append(element)

    return solutions

def process(search_space, len_sentence):

    # print(len(search_space))

    solutions = collections.defaultdict(list)
    nodes_map = {}
    reverse_nodes_map = {}

    G = nit.graph.Graph()
    for i, _ in search_space.items():
        new_node = G.addNode()
        nodes_map[i] = new_node
        reverse_nodes_map[new_node] = i


    number_of_connections = 0
    for i, cxn1 in search_space.items():
        for j, cxn2 in search_space.items():
            if j>i:
                if gutils.matchable(cxn1, cxn2):
                    number_of_connections += 1
                    G.addEdge(nodes_map[i], nodes_map[j])

    cliques = nit.clique.MaximalCliques(G).run()

    for element in cliques.getCliques():

        element = [reverse_nodes_map[i] for i in element]

        projected_sentence = ["_"]*len_sentence

        for idx in element:
            projected_sentence, _ = gutils.update(projected_sentence, search_space[idx])

        solutions[tuple(projected_sentence)].append(element)

    return solutions


def collapse_matrix(output_dir: Path, input_dir: Path, catenae_fpath: Path,
                    multiprocessing: bool, n_workers: int, chunksize: int) -> None:
    """_summary_

    Args:
        output_dir (Path): _description_
        input_dir (Path): _description_
        catenae_fpath (Path): _description_
        multiprocessing (bool): _description_
        n_workers (int): _description_
        chunksize (int): _description_
    """

    catenae_idx = {cxn:i for i, cxn in enumerate(dutils.load_catenae_set(catenae_fpath, math.inf))}
    with open(output_dir.joinpath("catenae.idx"), "w") as fout_catenae:
        for cxn, idx in catenae_idx.items():
            print(f"{idx}\t{cxn}", file=fout_catenae)

    filenames = input_dir.iterdir()

    if multiprocessing:
        process_map(functools.partial(collapse, output_dir=output_dir,
                                      input_dir=input_dir, catenae_idx=catenae_idx),
                    filenames, max_workers=n_workers, chunksize=chunksize)

        # with Pool(10) as p:
        #     ret = list(tqdm.tqdm(p.imap(functools.partial(basic_collapse, output_dir=output_dir, input_dir=input_dir),
        #                          filenames)))

    else:
        lstdir_it = tqdm.tqdm(filenames)
        with open(output_dir.joinpath("computing_times.txt"), "w") as fout_times:
            for filename in lstdir_it:
                filename = filename.name
                lstdir_it.set_description(f"Processing {filename}")
                start = time.time()
                collapse(filename, output_dir, input_dir, catenae_idx)
                end = time.time()
                print(f"{end-start}\t{filename}", file=fout_times)



def sentence_graph(output_dir: Path, input_dir: Path, catenae_fpath: Path,
                    multiprocessing: bool, n_workers: int) -> None:

    catenae_idx = {cxn:i for i, cxn in enumerate(dutils.load_catenae_set(catenae_fpath, math.inf))}
    with open(output_dir.joinpath("catenae.idx"), "w") as fout_catenae:
        for cxn, idx in catenae_idx.items():
            print(f"{idx}\t{cxn}", file=fout_catenae)

    filenames = input_dir.iterdir()

    if multiprocessing:
        process_map(functools.partial(compute_sentence_graph, output_dir=output_dir,
                                      catenae_idx=catenae_idx),
                    filenames, max_workers=n_workers)
    else:
        for filename in filenames:
            compute_sentence_graph(filename, output_dir, catenae_idx)



def compute_sentence_graph(filename: Path, output_dir: Path, catenae_idx: Dict[str, int]) -> None:

    basename = filename.name

    with open(filename) as fin:
        with open(output_dir.joinpath(f"{basename}.index"), "w") as fout_index:
            print("PRINTED\tSENTENCE\tNODES\tEDGES\tLINEAR", file=fout_index)

            fin_lines = fin.readlines()

            mat = []
            mat_count = 0

            for line in tqdm.tqdm(fin_lines):
                line = line.strip()

                if line:
                    mat.append(line.split("\t"))
                else:
                    number_of_connections = 0
                    mat_count_str = str(mat_count).zfill(4)

                    if mat:

                        rev_mat = [[x[i] for x in mat] for i in range(len(mat[0]))]
                        full_sentence = rev_mat[0]

                        if len(rev_mat) > 1:

                            full_sentence, search_space = rev_mat[0], rev_mat[1:]

                            search_space_dict = collections.defaultdict(list)
                            for cxn in search_space:
                                search_space_dict[catenae_idx[gutils.strip_underscore(cxn)]].append(cxn)

                            search_space_uniquedict = {}
                            for i, el_lst in search_space_dict.items():
                                if len(el_lst) == 1:
                                    search_space_uniquedict[str(i)] = el_lst[0]
                                else:
                                    idx = 1
                                    for el in el_lst:
                                        search_space_uniquedict[f"{str(i)}_{idx}"] = el
                                        idx += 1

                            prog_idx = 1
                            prog_idx_dict = {}
                            for dict_key in search_space_uniquedict:
                                # print(f"{prog_idx}\t{dict_key}", file=fout_map)
                                prog_idx_dict[dict_key] = prog_idx
                                prog_idx += 1

                            lst_to_print = []
                            for i, cxn1 in search_space_uniquedict.items():
                                for j, cxn2 in search_space_uniquedict.items():
                                    if j>i:
                                        if gutils.matchable(cxn1, cxn2):
                                            number_of_connections += 1
                                            lst_to_print.append(f"{prog_idx_dict[i]} {prog_idx_dict[j]}")



                            if number_of_connections > 0:
                                with open(output_dir.joinpath(f"{basename}_{mat_count_str}.mtx"), "w") as fout_matrix, \
                                open(output_dir.joinpath(f"{basename}_{mat_count_str}.map"), "w") as fout_map:

                                    print("GRAPH_IDX\tOVERALL_IDX\tCATENA", file=fout_map)
                                    for dict_key, prog_idx in prog_idx_dict.items():
                                        catena = search_space_uniquedict[dict_key]
                                        catena_str = " ".join(catena)
                                        print(f"{prog_idx}\t{dict_key}\t{catena_str}", file=fout_map)

                                    print("%%MatrixMarket matrix coordinate pattern symmetric", file=fout_matrix)
                                    print("%{}".format(" ".join(full_sentence)), file=fout_matrix)
                                    print(f"{prog_idx} {prog_idx} {number_of_connections}", file=fout_matrix)
                                    print("\n".join(lst_to_print), file=fout_matrix)

                        full_sentence = " ".join(full_sentence)
                        if number_of_connections > 0:
                            print(f"{1}\t{mat_count_str}\t{len(prog_idx_dict)}\t{number_of_connections}\t{full_sentence}", file=fout_index)
                        else:
                            print(f"{0}\t{mat_count_str}\t0\t0\t{full_sentence}", file=fout_index)

                        mat_count += 1
                        mat = []


def collapse(filename: Path, output_dir: Path, input_dir: Path,
             catenae_idx: Dict[str, int]) -> None:
    """_summary_

    Args:
        filename (Path): _description_
        output_dir (Path): _description_
        input_dir (Path): _description_
        catenae_fpath (Path): _description_
    """



    with open(input_dir.joinpath(filename)) as fin, \
        open(output_dir.joinpath(filename), "w") as fout_translation, \
            open(output_dir.joinpath(f"{filename}.cxns"), "w") as fout_cxns:

        fin_lines = fin.readlines()
        mat = []
        # lines_it = tqdm.tqdm(fin_lines)
        # lines_it.set_description(filename)
        for line in tqdm.tqdm(fin_lines):
            line = line.strip()

            if line:
                mat.append(line.split("\t"))
            else:
                if mat:
                    # print("\n".join("\t".join(el) for el in mat))
                    rev_mat = [[x[i] for x in mat] for i in range(len(mat[0]))]
#                    if len(rev_mat) > 1 and len(rev_mat) < 200: #TODO: check here

                    if len(rev_mat) > 1:

                        full_sentence, search_space = rev_mat[0], rev_mat[1:]
                        # print(" ".join(full_sentence))
                        len_sentence = len(full_sentence)
                        # search_space = [(not_underscore(el), el) for el in search_space]

                        # search_space.sort(key=operator.itemgetter(0))

                        # groups = itertools.groupby(search_space, key=operator.itemgetter(0))

                        # search_space = [[item for item in data] for (key, data) in groups]

                        search_space_dict = collections.defaultdict(list)
                        for cxn in search_space:
                            search_space_dict[catenae_idx[strip_underscore(cxn)]].append(cxn)
                        # search_space = {catenae_idx[strip_underscore(cxn)]: cxn
                        #                 for cxn in search_space}

                        search_space_uniquedict = {}
                        for i, el_lst in search_space_dict.items():
                            if len(el_lst) == 1:
                                search_space_uniquedict[str(i)] = el_lst[0]
                            else:
                                idx = 1
                                for el in el_lst:
                                    search_space_uniquedict[f"{str(i)}_{idx}"] = el
                                    idx += 1


                        # for el in search_space_uniquedict:
                        #     print(el, search_space_uniquedict[el])
                        # input()

                        solutions = process(search_space_uniquedict, len_sentence)



                        # print(solutions)

                        # scored_solutions = [(x, y, abscore(y), lenscore(y)) for x, y in solutions]
                        # sorted_solutions = sorted(scored_solutions, key=lambda x: (x[2], x[3]), reverse=True)

                        print("SENTENCE:", " ".join(full_sentence), file=fout_translation)
                        print("SENTENCE:", " ".join(full_sentence), file=fout_cxns)
                        # print("SENTENCE:", " ".join(rev_mat[0]))
                        # print("SENTENCE:", " ".join(rev_mat[0]))

                        print("\t".join(search_space_uniquedict.keys()), file=fout_cxns)

                        for solution, built_from_lst in solutions.items():
                            solution = gutils.add_lexical_items(full_sentence, list(solution))
                            to_print = " ".join(solution)

                            for built_from in built_from_lst:
                                built_from_str = " ".join(built_from)
                                to_print += f"\t{built_from_str}"

                            print(to_print, file=fout_translation)
                            # print(" ".join(solution),
                            #       "\t", built_from_str, file=fout_translation)
                        print("\n", file=fout_cxns)
                        print("\n", file=fout_translation)

                        # for built_from, solution, abscore_v, lenscore_v in sorted_solutions:
                        #     built_from_str = " ".join(str(x) for x in built_from)

                        #     solution = add_lexical_items(full_sentence, list(solution))
                        #     print("\t", f"{abscore_v:.2f}", "\t", lenscore_v,
                        #         "\t\t", " ".join(solution),
                        #         "\t", built_from_str, file=fout_translation)
                            # print("\t", f"{abscore_v:.2f}", "\t", lenscore_v,
                                # "\t\t", " ".join(solution),
                                # "\t", built_from_str)
                mat = []


def find_cliques(output_dir: Path, input_dir: Path, bin_size: int, batch_size: int) -> None:
    """_summary_

    Args:
        output_dir (Path): _description_
        input_dir (Path): _description_
        bin_size (int): _description_
    """

    min_nodes, max_nodes = 1000, 0
    min_edges, max_edges = 1000, 0
    index_files_it = tqdm.tqdm(input_dir.glob("*.index"),
                               bar_format="{desc:60.60s}: {percentage:3.0f}%|{bar}{r_bar}")
    files_number = 0
    logger.info("Finding min and max nodes")
    for filename in index_files_it:
        files_number += 1
        basename = filename.name
        index_files_it.set_description(basename)
        with open(filename) as fin:
            header = fin.readline()

            for line in fin:
                line = line.strip().split("\t")

                flag, idx, nodes, edges, sentence = line
                flag = int(flag)
                nodes = int(nodes)
                edges = int(edges)

                if flag == 1:
                    if nodes < min_nodes:
                        min_nodes = nodes
                    if nodes > max_nodes:
                        max_nodes = nodes
                    if edges < min_edges:
                        min_edges = edges
                    if edges > max_edges:
                        max_edges = edges

    # print(min_nodes, max_nodes)
    # print(min_edges, max_edges)

    bins = [[] for _ in range((((max_nodes-min_nodes)//bin_size)+1)) ]

    index_files_it = tqdm.tqdm(input_dir.glob("*.index"), total=files_number,
                               bar_format="{desc:60.60s}: {percentage:3.0f}%|{bar}{r_bar}")
    logger.info("Splitting sentences into bins")
    for filename in index_files_it:
        basename = filename.name
        index_files_it.set_description(basename)
        rootname = basename.split(".")[0]

        with open(filename) as fin:
            header = fin.readline()

            for line in fin:
                line = line.strip().split("\t")

                flag, idx, nodes, edges, sentence = line
                flag = int(flag)
                nodes = int(nodes)
                edges = int(edges)
                if flag == 1:
                    idx_bin = (nodes-min_nodes)//bin_size
                    bins[idx_bin].append((rootname, idx, nodes, edges))

    # catenae_idx = {cxn:i for i, cxn in enumerate(dutils.load_catenae_set(input_dir.joinpath("catenae.idx"), math.inf))}
    # reverse_catenae_idx = {i:cxn for }

    for i, bin in enumerate(bins):
        logger.info(f"sentences containing {(i*bin_size)+min_nodes} TO {((i+1)*bin_size)+min_nodes-1} nodes: {len(bin)}")


    with open(output_dir.joinpath("computing_times.report"), "w") as fout_report:
        for i, bin in enumerate(bins): #TODO: restrict computation to only some bins

            print(f"# SENTENCES CONTAINING {(i*bin_size)+min_nodes} TO {((i+1)*bin_size)+min_nodes-1} NODES", file=fout_report)
            logger.info(f"projecting sentences containing {(i*bin_size)+min_nodes} TO {((i+1)*bin_size)+min_nodes-1} nodes")

            printed_sentences = 0
            batch = 0
            batch_str = str(batch).zfill(5)
            fout_sentences = open(output_dir.joinpath(f"nodes_{(i*bin_size)+min_nodes}_to_{((i+1)*bin_size)+min_nodes-1}_batch{batch_str}"), "w")

            for rootname, idx, nodes, edges in tqdm.tqdm(bin):
                start = time.time()

                map_path = input_dir.joinpath(f"{rootname}_{idx}.map")
                mtx_path = input_dir.joinpath(f"{rootname}_{idx}.mtx")

                sentence, solutions = project_sentence(output_dir, rootname, idx, map_path, mtx_path)

                print(f"%{rootname} {idx}", file=fout_sentences)
                print(sentence, file=fout_sentences)
                for solution, cliques in solutions.items():
                    str_to_print = " ".join(solution)
                    for clique in cliques:
                        str_to_print += "\t"+" ".join(str(i) for i in clique)
                    print(str_to_print, file=fout_sentences)
                print("\n", file=fout_sentences)

                end = time.time()
                elapsed_time = end-start

                printed_sentences += 1
                if printed_sentences == batch_size:
                    fout_sentences.close()
                    printed_sentences = 0
                    batch += 1
                    batch_str = str(batch).zfill(5)
                    fout_sentences = open(output_dir.joinpath(f"nodes_{(i*bin_size)+min_nodes}_to_{((i+1)*bin_size)+min_nodes-1}_batch{batch_str}"), "w")


                print(f"{rootname}\t{idx}\t{nodes}\t{edges}\t{elapsed_time}", file=fout_report)

            print("\n", file=fout_report)

            fout_sentences.close()


def project_sentence(output_dir: Path, rootname: str, idx: str,
                     map_path: Path, mtx_path: Path) -> List[str]:
    """_summary_

    Args:
        output_dir (_type_): _description_
        rootname (_type_): _description_
        idx (_type_): _description_
        map_path (_type_): _description_
        mtx_path (_type_): _description_
    """
    len_sentence = 0
    G = nit.graph.Graph()
    nodes_map = {}
    rev_nodes_map = {}
    cxn_map = {}
    with open(map_path) as fin:
        fin.readline()

        for line in fin:
            graph_idx, global_idx, cxn = line.strip().split("\t")
            graph_idx = int(graph_idx)
            cxn = cxn.split(" ")
            len_sentence = len(cxn)

            new_node = G.addNode()
            nodes_map[graph_idx] = new_node
            rev_nodes_map[new_node] = graph_idx
            cxn_map[new_node] = cxn

    with open(mtx_path) as fin:
        fin.readline()
        sentence = fin.readline().strip()
        fin.readline()

        for line in fin:
            i, j = line.strip().split(" ")
            i, j = int(i), int(j)
            G.addEdge(nodes_map[i], nodes_map[j])

    # logger.info("graph loaded")
    cliques = nit.clique.MaximalCliques(G).run()
    # logger.info("cliques computed")
    solutions = collections.defaultdict(list)

    for element in cliques.getCliques():
        # print(element)
        rev_element = [rev_nodes_map[i] for i in element]

        cxn_element = [cxn_map[i] for i in element]

        projected_sentence = ["_"]*len_sentence

        for cxn in cxn_element:
            projected_sentence, _ = gutils.update(projected_sentence, cxn)

        solutions[tuple(projected_sentence)].append(rev_element)

    return sentence, solutions

# TODO: switch to tqdm multiprocess""