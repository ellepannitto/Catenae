import pandas as pd

import sys

fin = sys.argv[1]

dataframe = pd.read_csv(fin, sep='\t', header=0)

print(dataframe.head)