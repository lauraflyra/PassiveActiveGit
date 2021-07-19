import numpy as np
import pandas as pd

corpus_df = pd.read_csv("../incomplete_corpus_data/PotsdamSearchCorpus.dat", sep = "\t")
#testing with raw data from participant 1
raw_data = pd.read_csv("../incomplete_corpus_data/asc\SFC1_2.dat", sep = "\t",
                       header = None)
