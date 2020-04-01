import pandas as pd
import re
import itertools
import numpy as np
import pandas as pd

from collections import Counter
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

from nltk import everygrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer


df = pd.read_excel('data/OperationeleRapportage_25-02-2020_10-03-2020.xlsx')

stemmer = SnowballStemmer("dutch")
textdata = [word_tokenize(str(x)) for x in df['Eigen Input']]

# stem words
textdata = [ [stemmer.stem(x) for x in sentence] for sentence in textdata]

word_counts = Counter(itertools.chain(*textdata))
# most common words (no stop words)
most_common_words = [x[0] for x in word_counts.most_common() if x[0] not in stopwords.words('dutch')]
vocabulary = {key: value for value, key in enumerate(most_common_words)}

print(most_common_words[:25])


gramdata = [[" ".join(x) for x in everygrams(str(sent).split(), min_len=2, max_len=3)] for sent in df['Eigen Input']]
ngram_counts = Counter(itertools.chain(*gramdata))
most_common_ngrams = [x[0] for x in word_counts.most_common() if x[0] not in stopwords.words('dutch')]

print(most_common_words[:25])
