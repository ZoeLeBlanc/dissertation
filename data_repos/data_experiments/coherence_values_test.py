# import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import MDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_selection import chi2
import os, sys
import glob
import gensim
from gensim.corpora import Dictionary
from gensim.similarities import MatrixSimilarity
from gensim.models import ldamodel, doc2vec, LsiModel, LdaModel, CoherenceModel
from gensim.matutils import kullback_leibler, jaccard, hellinger, sparse2full
import nltk
# nltk.download('punkt')
import string
import csv
import math
import statistics
import datetime
from nltk.corpus import stopwords
from nltk.util import ngrams
# nltk.download('stopwords')
from collections import OrderedDict, Counter, namedtuple
import random
import codecs, difflib, distance
import rpy2

from progress.bar import IncrementalBar
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt


from ast import literal_eval
import itertools
# %load_ext rpy2.ipython
df = pd.read_csv('../scripts/combined_pages_date_congo_text_ner_binned.csv')
print(len(df))
model_texts = []
def get_texts(rows):
    
    x = literal_eval(rows.values[0])
    t = ' '.join(x)
    model_texts.append(t)
    return t

df_grouped = df.groupby(['date', 'binned'])['token_texts'].apply(get_texts).reset_index()

model_lists = []
def get_lists(rows):

    x = literal_eval(rows.values[0])
#     print(len(x))
    combined = list(itertools.chain.from_iterable(x))

#     print(len(combined))
    model_lists.append(combined)
#     print(len(model_lists))
    return combined

df_grouped1 = df.groupby(['date', 'binned'])['token_lists'].apply(get_lists).reset_index()
print(len(model_texts))

dictionary_zero = Dictionary(model_lists)
dictionary_zero.filter_extremes( no_above=0.2)
corpus_zero = [dictionary_zero.doc2bow(text) for text in model_lists]

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print(len(model_list))
        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, iterations=400, alpha='auto', eta='auto', passes=20)
        model_list.append(model)
        print(len(model_list))
        print(len(coherence_values))
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print(len(coherence_values))

    return model_list, coherence_values
limit=100; start=2; step=5;
x = range(start, limit, step)
model_list, coherence_values = compute_coherence_values(dictionary=dictionary_zero, corpus=corpus_zero, texts=model_lists, start=start, limit=limit, step=step)
# # # # Show graph

plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.savefig('coherence_topic_model_congo_100.png')
plt.show()
