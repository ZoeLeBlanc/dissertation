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

def process_text(df, types):
    doc = []
    final_doc = []
    for index, row in df.iterrows():
        raw_text = row['lowercase']
        tokens = custom_tokenize(raw_text)
        page_terms = ''
        for t in tokens:
            if t in string.punctuation:
                pass
            elif t in stopwords.words('english'):
                pass
            else:
                page_terms += t.lower() + ' '
        doc.append(page_terms)

    for sent in doc:
        sent_terms = ''
        spacy_text = nlp(sent)
        for ent in spacy_text.ents:
            if ent.label_ in types:
                if '.' in ent.text:
                    text = ('').join(ent.text.split('.'))
                    sent_terms += text + ' '
                else:
                    sent_terms += ent.text + ' '
        final_doc.append(sent_terms)
    return create_matrix(final_doc)
    
def create_matrix(ents):
    count_model = CountVectorizer(ngram_range=(2,2)) # default unigram model
    X = count_model.fit_transform(ents)
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
    return df

def get_files(dir, output_path):
    os.chdir(dir)
    files = [i for i in glob.glob('*.{}'.format('csv')) if 'grouped' in i and 'no' not in i]
    files.sort()
    print(files)
    # # final_df = pd.DataFrame(columns=['page', 'lowercase', 'counts'], index=None)
    # output_path = 'final_htrc.csv'
    ner_data = IncrementalBar('ner data for volume', max=len(files))
    for filename in files:
        ner_data.next()
        htrc_df = pd.read_csv(filename, index_col=False)
        types = ['LOC', 'GPE']
        df = process_text(htrc_df, types)
        file_name = filename.split('.')[0]
        df['htrc_vol'] = file_name
        print(df)
        if os.path.exists(output_path):
            df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            df.to_csv(output_path, header=True, index=False)
    ner_data.finish()

if __name__ ==  "__main__" :
	# print(sys.argv[1])
	get_files('./Arab_Observer_HTRC', 'bigram_arab_observer_ner.csv')
    # get_files(sys.argv[1], sys.argv[2])

