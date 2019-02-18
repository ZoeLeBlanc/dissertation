import spacy
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
from gensim.models import ldamodel, doc2vec, LsiModel 
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
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.layouts import row, column
from progress.bar import IncrementalBar
import warnings
warnings.filterwarnings('ignore')
nlp = spacy.load('en_core_web_lg')
# %load_ext rpy2.ipython

def custom_tokenize(text):
    if not text:
#       print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    return nltk.word_tokenize(text)

def process_text(data, term):

    data = pd.read_csv(data)
    if term: 
        data = data[data['tokenized'].str.contains(term) == True]
    print(data.columns)
    create_matrix(data[0:100])
    
def create_matrix(ents):
    count_model = CountVectorizer(ngram_range=(1,3)) # default unigram model
    X = count_model.fit_transform(ents.terms)
    Xc = (X.T * X)
    # Xc.setdiag(0)
    vocab = count_model.vocabulary_
    vocab2 = {y:x for x,y in vocab.items()}
    return create_network(Xc, vocab2)
    
def create_network(matrix, vocab):
    G = nx.from_scipy_sparse_matrix(matrix)
    H = nx.relabel_nodes(G, vocab)
    df = nx.to_pandas_edgelist(H)
    df = df[df['source'].str.isnumeric() == False]
    df = df[df['target'].str.isnumeric() == False]
    df['terms'] = df['source'] + '_' + df['target']
    # df = df.drop(columns = ['source', 'target'])
    print(df)
    return df


if __name__ ==  "__main__" :
	# print(sys.argv[1])
    df = process_text('./Arab_Observer_HTRC/full_arab_observer_ner_test.csv', '')
    df.to_csv('test_places_occur.csv')
    # get_files(sys.argv[1], sys.argv[2])

