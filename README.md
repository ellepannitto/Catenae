# Catenae

## How to install

1. Create a virtual environment dedicated to the project
   
        `python3.7 -m venv venv_name`

2. Activate `venv`

        `source venv_name/bin/activate`
   
3. Install repo, from  `Catenae` main folder:

        `python3.7 setup.py install`

4. Install requirements

        `pip install -r requirements.txt`


## Usage

    usage: catenae [-h] {extract,weight,filter,cooccurrences,build-dsm}

### `extract`

The script extracts list of catenae from a parsed input.

    catenae extract [-o OUTPUT_DIR] 
                     [-c CORPUS_DIRPATH]
                     [-m MIN_LEN_SENTENCE] 
                     [-M MAX_LEN_SENTENCE]
                     [-b SENTENCES_BATCH_SIZE] 
                     [-f FREQUENCY_THRESHOLD]
                     [--min-len-catena MIN_LEN_CATENA]
                     [--max-len-catena MAX_LEN_CATENA]


Here is a working example:
    
    catenae extract -o data/results/
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


| **IMPORTANT**: |
| -------------- |
| make sure that the output folder is empty before running the command |


### `weight`

The script computes a weight function (MI) over a list of catenae 
and given the files created at the `extract` step.

    catenae weight [-o OUTPUT_DIR] 
                   -i ITEMS_FILEPATH 
                   -t TOTALS_FILEPATH 
                   -c CATENAE_FILEPATH

Here is a working example:

    catenae weight -o data/results/
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

### `filter`

The script filters the weighted list of catenae on frequency 
and weight.

The employed weighting function is a generalized version of PMI.


    catenae filter [-o OUTPUT_DIR] 
                   -i INPUT_FILEPATH 
                   [-f MIN_FREQ]
                   [-w MIN_WEIGHT]
                   [-m MIN_LEN_CATENA]
                   [-M MAX_LEN_CATENA]


Here is a working example:

    catenae filter -o data/results/ 
                   -i data/results/catenae-weighted.txt 
                   -f 100 
                   -w 0 
                   -m 0 
                   -M 3

The script will produce a file named `catenae-filtered.txt`, 
formatted like `catenae-weighted.gz` but containing catenae with
minimum frequency of 100 (`-f`), positive mutual information (`-w`),
and length between 0 and 3 (`-m` and `-M`).

### `cooccurrences`

The script extracts co-occurrences of catenae, to be used for 
building the distributional space.


    catenae filter [-o OUTPUT_DIR] 
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

    catenae cooccurrences -o data/cooccurrences/ 
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

| **IMPORTANT**: |
| -------------- |
| make sure that the output folder is empty before running the command |

### `build-dsm`

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
This step will produce two files:
* `catenae-ppmi.gz` containing a weighted version of raw cooccurrences
* `catenae-dsm.gz` containing the implicit vectors reduced to 300 dimensions

## sample-input

The script samples a `train`, `development` and `test` set from a set of input files.

    catenae sample-input [-o OUTPUT_DIR] 
                         -c CORPUS_DIRPAHT 
                         -s SIZE
                         --seed RANDOM_SEED

Here is a working example:

    catenae sample-input -o data/input_sampled/ 
                         -c data/input_test/ 
                         -s 3000

The above command will create `train`, `valid` and `test` files in the `data/input_sampled/` 
folder, both in linear version (`.txt` extension) and parsed version (`.conll` extension).