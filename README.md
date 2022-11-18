# Catenae

## How to install

1. Create a virtual environment dedicated to the project

        `python3.7 -m venv venv_name`

2. Activate `venv`

        `source venv_name/bin/activate`

3. Install repo, from  `Catenae` main folder:

        `python3.7 setup.py [develop|install]`

4. Install requirements

        `pip install -r requirements.txt`


## Usage

    usage: catenae [-h]
      {pos-stats,morph-stats,verbedges-stats,subj-stats,synrel-stats,extract-catenae,weight-catenae,filter-catenae,extract-cooccurrences,build-dsm,sample-input,udparse,spearman,corecatenae,extract-sentences,similarity-matrix,reduce-simmatrix,query-neighbors,glassify-matrix,glassify-collapse}

---

## Options:

### Statistics on corpus

#### `pos-stats`

#### `morph-stats`

#### `verbedges-stats`

#### `sbj-stats`

#### `synrel-stats`

---

### From corpus to catenae

#### `udparse` -- IT WORKS!

The script parses a linear version of the corpus using Universal Dependencies formalism.

    catenae udparse [-o OUTPUT_DIR]
                    -i INPUT_DIR
                    -m MODEL

Here is a working example:

    catenae udparse -o data/corpus_parsed
                    -i data/linear_corpus
                    -m data/udpipe_models/english-ewt-ud-2.3-181115.udpipe

The file loads a UDPipe model and parses all (tokenized) input files into the `.conllu` format.

Example of input format:

    MOT	what 's that
    MOT	it 's a chicken
    CHI	yeah
    MOT	yeah
    MOT	what 's this

Corresponding parsed format provided in output:

(Note: labels as MOT and CHI are removed in working version of the script. They've been kept here just for sake of simplicity)

    # sent_id = 3
    1	MOT	Mot	CCONJ	CC	_	2	cc	_	_
    2	what	what	PRON	WP	PronType=Int	3	nsubj	_	_
    3	's	be	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	_	_
    4	that	that	PRON	DT	Number=Sing|PronType=Dem	3	nsubj	_	_

    # sent_id = 4
    1	MOT	Mot	CCONJ	CC	_	5	cc	_	_
    2	it	it	PRON	PRP	Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs	5	nsubj	_	_
    3	's	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	5	cop	_	_
    4	a	a	DET	DT	Definite=Ind|PronType=Art	5	det	_	_
    5	chicken	chicken	NOUN	NN	Number=Sing	0	root	_	_

    # sent_id = 5
    1	CHI	chi	INTJ	UH	_	0	root	_	_
    2	yeah	yeah	INTJ	UH	_	1	discourse	_	_

    # sent_id = 6
    1	MOT	Mot	PART	RB	_	0	root	_	_
    2	yeah	yeah	INTJ	UH	_	1	discourse	_	_

    # sent_id = 7
    1	MOT	Mot	CCONJ	CC	_	2	cc	_	_
    2	what	what	PRON	WP	PronType=Int	0	root	_	_
    3	's	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	2	cop	_	_
    4	this	this	PRON	DT	Number=Sing|PronType=Dem	2	nsubj	_	_


#### `extract-catenae` -- IT WORKS!

The script extracts list of catenae from a parsed input.

    catenae extract-catenae [-o OUTPUT_DIR]
                            [-c CORPUS_DIRPATH]
                            [-m MIN_LEN_SENTENCE]
                            [-M MAX_LEN_SENTENCE]
                            [-b SENTENCES_BATCH_SIZE]
                            [-f FREQUENCY_THRESHOLD]
                            [--min-len-catena MIN_LEN_CATENA]
                            [--max-len-catena MAX_LEN_CATENA]


Here is a working example:

    catenae extract-catenae -o data/results/
                            -c data/corpus/
                            -m 2
                            -M 25
                            -f 3


The above command will extract catenae of length between 0
(`--min-len-catena`) and 5 (`--max-len-catena`)
(default parameters) from the parsed files contained in `data/corpus/`.
Only sentences of length between 2 (`-m`) and 25 (`-M`) will be
considered, in batches of 100 (`-b`). For each batch, only catenae
with frequencies higher than 3 (`-f`) will be kept.

The script will produce three gzipped files in `data/results/`:
* `catenae-freq-summed.gz` containing catenae in alphabetical order and
  their frequencies, tab separated

* `items-freq-summed.gz` containing items (including words, syntactic
  relations and pos tags) and their overall frequencies, tab separated

* `totals-freq-summed.gz` containing the overall frequencies for catenae
of each length, and the total words count for the corpus



#### `weight-catenae` -- IT WORKS!

The script computes a weight function (MI) over a list of catenae
and given the files created at the `extract` step.

    catenae weight-catenae [-o OUTPUT_DIR]
                           -i ITEMS_FILEPATH
                           -t TOTALS_FILEPATH
                           -c CATENAE_FILEPATH

Here is a working example:

    catenae weight-catenae -o data/results/
                           -i data/results/items-freq-summed.gz
                           -t data/results/totals-freq-summed.gz
                           -c data/output_test/catenae-freq-summed.gz

The script will produce a file named `catenae-weighted.gz` in the output
folder.
The file has three tab-separated columns, formatted as follows:

| CATENA | FREQ | W |
| ----- | ---- | --- |
| _VERB @case @obl | 978.0 | 1617.3239266779776 |
|_VERB _ADP @obl |   1005.0|  1415.5696170807614 |
|@case @obl      |   1047.0 | 1247.283448889141 |
|_ADP @obl       |   1041.0 | 935.3736059263747 |
| ... | ... | ... |

#### `filter-catenae` -- IT WORKS!

The script filters the weighted list of catenae on frequency
and weight.

The employed weighting function is a generalized version of PMI.


    catenae filter-catenae [-o OUTPUT_DIR]
                           -i INPUT_FILEPATH
                           [-f MIN_FREQ]
                           [-w MIN_WEIGHT]
                           [-m MIN_LEN_CATENA]
                           [-M MAX_LEN_CATENA]


Here is a working example:

    catenae filter-catenae -o data/results/
                           -i data/results/catenae-weighted.txt
                           -f 100
                           -w 0
                           -m 0
                           -M 3

The script will produce a file named `catenae-filtered.txt`,
formatted like `catenae-weighted.gz` but containing catenae with
minimum frequency of 100 (`-f`), positive mutual information (`-w`),
and length between 0 and 3 (`-m` and `-M`).

---

### Preparing for babbling

#### `sample-input` -- IT WORKS!

The script samples a `train`, `development` and `test` set from a set of input files.

    catenae sample-input [-o OUTPUT_DIR]
                         -c CORPUS_DIRPAHT
                         -s SIZE
                         --seed RANDOM_SEED

Here is a working example:

    catenae sample-input -o data/output_sampled/
                         -c data/corpus/
                         -s 800
                         --seed 132

The above command will create `train`, `valid` and `test` files in the `data/input_sampled/`
folder, both in linear version (`.txt` extension) and parsed version (`.conll` extension).

---

### Distributional model

#### `extract-cooccurrences` -- IT WORKS!

The script extracts co-occurrences of catenae, to be used for
building the distributional space.


    catenae extract-cooccurrences [-o OUTPUT_DIR]
                                  [-c CORPUS_DIRPATH]
                                  -a ACCEPTED_CATENAE
                                  [-k TOP_K]
                                  [-m MIN_LEN_SENTENCE]
                                  [-M MAX_LEN_SENTENCE]
                                  [-b SENTENCES_BATCH_SIZE]
                                  [-f FREQUENCY_THRESHOLD]
                                  [--min-len-catena MIN_LEN_CATENA]
                                  [--max-len-catena MAX_LEN_CATENA]
                                  [--include-len-one-items]
                                  --words-filepath WORDS_FILEPATH

Here is a working example:

    catenae extract-cooccurrences -o data/cooccurrences/
                                  -c data/corpus/
                                  -a data/results/catenae-filtered.txt
                                  -m 3
                                  -b 5000
                                  -f 1
                                  --include-len-one-items
                                  --words-filepath data/results/items-freq-summed.gz

The above command will extract cooccurrences between the catenae contained in
`catenae-filtered.txt` and items of length one contained in `items-freq-summed.gz`
file produced at the `extract` step, as the flag `--include-len-one-items` is present.
It will only look for co-occurrences in sentences of length between `3` (`-m`) and
`25` (`-M`, default), in batches of `5000` (`-b`).

The script will produce three gzipped files in `data/cooccurrences/`:
* `catenae-coocc-summed.gz` containing pairs of catenae in alphabetical order and
  their cooccurrence frequency, tab separated

* `catenae-freqs-summed.gz` containing catenae in alphabetical order and
  their frequencies, tab separated

* `totals-freqs.txt` containing total count of items in the corpus
  (used in the next step to compute MI)


#### `build-dsm` -- IT WORKS!

The script builds the distributional semantic space, given the
files created in the `cooccurrences` step.

    catenae build-dsm [-o OUTPUT_DIR]
                      -c COOCCURRENCES_FILEPATH
                      -f FREQUENCIES_FILEPATH
                      -t TOTAL

Here is a working example:

    catenae build-dsm -o data/output_dsm/
                      -c data/cooccurrences/catenae-coocc-summed.gz
                      -f data/cooccurrences/catenae-freqs-summed.gz
                      -t 68413568

The above command will create a distributional space based on cooccurrences
extracted during the previous step. In particular, files `catenae-coocc-summed.gz`
and `catenae-freqs-summed.gz` are those created by the `cooccurrences` command,
while the integer to be used as `-t` parameter is to be found in the `totals-freqs.txt`
file also created during the previous step.
This step will produce three files:
* `catenae-ppmi.gz` containing a weighted version of raw cooccurrences
* `catenae-dsm.idx.gz` containing the vocabulary for the implicit vectors reduced to 300 dimensions
* `catenae-dsm.vec.gz` containing the implicit vectors reduced to 300 dimensions


#### `similarity-matrix` -- IT WORKS!

The script computes a matrix containing **cosine similarities** between pairs of vectors in the
distributional model built with the command `build-dsm`.

As computing the full similarity matrix is potentially both highly space and time consuming,
the command allows for multiple options:
* it is possible to specify the subset of vectors we want to consider for computing similarity
* both a `full` and a `chunked` version are available. The chunked version is slower but memory-efficient.

      catenae similarity-matrix [-o OUTPUT_DIR]
                                -s DSM_VEC
                                -i DSM_IDX
                                [--reduced-left-matrix]
                                [--reduced-right-matrix]
                                [--chunked]
                                [--working-memory]

Here is a working example:

      catenae similarity-matrix -o data/output_simmatrix/
                                -s data/output_dsm/catenae-dsm.vec.gz
                                -i data/output_dsm/catenae-dsm.idx.gz
                                --reduced-left-matrix data/catenae_subsets/left_catenae.txt
                                --reduced-right-matrix data/catenae_subsets/right_catenae.txt
                                --chunked
                                --working-memory 2000

| IMPORTANT |
-----
The catenae contained in the files used as `reduced-left-matrix` and `reduced-right-matrix` should be subsets of the catenae contained in the file `DSM_IDX`. It is not mandatory that the order in which they appear in the file is the same, nonetheless there cannot be catenae in `reduced-[left\|right]-matrix` file that do not appear in the `DSM_IDX` file.

The above command will create various files in the designated output folder.
More specifically:
* `idxs.left` and `idxs.right`, in case the options `--reduced-[left|right]-matrix` are used, containing the sorted list of catenae, each one corresponding to a row or column in the similarity matrix (`idxs.left` for row indexes, `idxs.right` for column indexes).

* If the `--chunked` option is provided, a number of `simmatrix.*.npy` files are produced, each one containing a chunk of the similarity matrix in numpy format.

* If the `--chunked` option is not provided, a `single simmatrix.npy` file is, containing the entire similarity matrix.

#### `reduce-matrix`

#### `query-neighbors`

---

## Investigating community of speakers

#### `spearmanr`

#### `corecatenae`

#### `extract-sentences`

#### `glassify-matrix`

#### `glassify-collapse`