import sys
import catenae


output_dir_basename = sys.argv[1]
corpus_dir_basename = sys.argv[2]
weighted_dir_basename = sys.argv[3]


for i in range(1, 11):
    n = str(i).zfill(2)

    output_dir = output_dir_basename+"{}/".format(n)
    corpus_dir = corpus_dir+"{}/".format(n)

    accepted_catenae_filepath = weighted_dir_basename+"{}/catenae-filtered.txt".format(n)

    top_k = float("inf")
    min_len_sentence = 3
    max_len_sentence = 20

    sentences_batch_size = 5000

    min_freq = 2

    min_len_catena = 0
    max_len_catena = 3

    include_words = True

    words_filepath = weighted_dir_basename+"{}/items-freq-summed.gz".format(n)





extraction.extract_coccurrences(output_dir, corpus_dir, accepted_catenae_filepath, top_k,
                                    min_len_sentence, max_len_sentence, sentences_batch_size,
                                    min_freq, min_len_catena, max_len_catena,
                                    include_words, words_filepath)



#    parser_coocc.add_argument("-o", "--output-dir", default="data/results/",
#                              help="path to output dir, default is data/results/")
#    parser_coocc.add_argument("-c", "--corpus-dirpath", default="data/corpus/",
#                              help="path to corpus directory")
#    parser_coocc.add_argument("-a", "--accepted-catenae",
#                              help="filepth containing accepted catenae")
#    parser_coocc.add_argument("-k", "--topk", type=float, default=float("inf"),
#                              help="number of catenae to load")
#    parser_coocc.add_argument("-m", "--min-len-sentence", type=int, default=1,
#                              help="minimum length for sentences to be considered")
#    parser_coocc.add_argument("-M", "--max-len-sentence", type=int, default=25,
#                              help="maximum length for sentences to be considered")
#    parser_coocc.add_argument("-b", "--sentences-batch-size", type=int, default=100,
#                              help="number of sentences in batch when extracting catenae")
#    parser_coocc.add_argument("-f", "--frequency-threshold", type=int, default=1,
#                              help="frequency threshold applied in each batch for a catena to be kept")
"""     parser_coocc.add_argument("--min-len-catena", type=int, default=0,
                              help="minimium length for each catena to be kept")
    parser_coocc.add_argument("--max-len-catena", type=int, default=3,
                              help="maximum length for each catena to be kept. WARNING: highly impacts ram usage")
    parser_coocc.add_argument("--include-len-one-items", action='store_true',
                              help="include words regardless of their frequency and weight values")
    parser_coocc.add_argument("--words-filepath",
                              help="filepath to words list, to include if len one items are required")


catenae cooccurrences -o data/cooccurrences/ 
                      -c data/corpus/ 
                      -a data/results/catenae-filtered.txt 
                      -m 3 
                      -b 5000 
                      -f 1 
                      --include-len-one-items 
                      --words-filepath data/results/items-freq-summed.gz




-o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        path to output dir, default is data/results/ (default:
                        data/results/)
  -c CORPUS_DIRPATH, --corpus-dirpath CORPUS_DIRPATH
                        path to corpus directory (default: data/corpus/)
  -a ACCEPTED_CATENAE, --accepted-catenae ACCEPTED_CATENAE
                        filepth containing accepted catenae (default: None)
  -k TOPK, --topk TOPK  number of catenae to load (default: inf)
  -m MIN_LEN_SENTENCE, --min-len-sentence MIN_LEN_SENTENCE
                        minimum length for sentences to be considered
                        (default: 1)
  -M MAX_LEN_SENTENCE, --max-len-sentence MAX_LEN_SENTENCE
                        maximum length for sentences to be considered
                        (default: 25)
  -b SENTENCES_BATCH_SIZE, --sentences-batch-size SENTENCES_BATCH_SIZE
                        number of sentences in batch when extracting catenae
                        (default: 100)
  -f FREQUENCY_THRESHOLD, --frequency-threshold FREQUENCY_THRESHOLD
                        frequency threshold applied in each batch for a catena
                        to be kept (default: 1)
  --min-len-catena MIN_LEN_CATENA
                        minimium length for each catena to be kept (default:
                        0)
  --max-len-catena MAX_LEN_CATENA
                        maximum length for each catena to be kept. WARNING:
                        highly impacts ram usage (default: 3)
  --include-len-one-items
                        include words regardless of their frequency and weight
                        values (default: False)
  --words-filepath WORDS_FILEPATH
                        filepath to words list, to include if len one items
                        are required (default: None) """