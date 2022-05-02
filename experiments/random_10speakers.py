import random
import catenae
import sys

input_dir = sys.argv[1]


#500k
random.seed(15)

for i in range(1, 11):
    pseudo_random_seed = random.randint(0,1000)
    n = str(i).zfill(2)
    print(n)

    catenae.core.corpus.sample("../data/10x500k/{}/".format(n), input_dir, 500000, pseudo_random_seed)


#1m
random.seed(37)

for i in range(1, 11):
    pseudo_random_seed = random.randint(0,1000)
    n = str(i).zfill(2)
    print(n)

    catenae.core.corpus.sample("../data/10x1m/{}/".format(n), input_dir, 1000000, pseudo_random_seed)


#3m
random.seed(89)

for i in range(1, 11):
    pseudo_random_seed = random.randint(0,1000)
    n = str(i).zfill(2)
    print(n)

    catenae.core.corpus.sample("../data/10x3m/{}/".format(n), input_dir, 3000000, pseudo_random_seed)