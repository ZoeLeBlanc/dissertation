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
    print(type(text))
    if not text:
#       print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    return nltk.word_tokenize(text)

def process_text(df, types):
    doc = []
    final_doc = []
    
    df['tokenized'] = ''
    df['ner_tokens'] = ''
    raw_text = df.at[0, 'google_vision_text']
    tokens = custom_tokenize(raw_text)
    page_terms = ''
    for t in tokens:
        if t in string.punctuation:
            continue
        elif t in stopwords.words('english'):
            continue
        else:
            page_terms += t.lower() + ' '
    df.at[0,'tokenized'] = page_terms
    sent_terms = ''
    spacy_text = nlp(page_terms)
    for ent in spacy_text.ents:
        if ent.label_ in types:
            if '.' in ent.text:
                text = ('').join(ent.text.split('.'))
                sent_terms += text + ' '
            else:
                sent_terms += ent.text + ' '
    df.at[0,'ner_tokens'] = sent_terms
                
    # df_1 = df.loc[df.ner_tokens.str.len() > 0]
    sent_terms = sent_terms.split(' ')
    sent_terms = [ t for t in sent_terms if t != '']
    print(sent_terms)
    if (len(sent_terms) > 1):
        return create_matrix(df)
    else:
        return pd.DataFrame()
    
def create_matrix(df):
    print(df.ner_tokens)
    count_model = CountVectorizer(ngram_range=(1,1)) # default unigram model
    X = count_model.fit_transform(df.ner_tokens)
    Xc = (X.T * X)
    # Xc.setdiag(0)
    vocab = count_model.vocabulary_
    vocab2 = {y:x for x,y in vocab.items()}
    return create_network(Xc, vocab2, df)
    
def create_network(matrix, vocab, df):
    G = nx.from_scipy_sparse_matrix(matrix)
    H = nx.relabel_nodes(G, vocab)
    df_1 = nx.to_pandas_edgelist(H)
    df_1 = df_1[df_1['source'].str.isnumeric() == False]
    df_1 = df_1[df_1['target'].str.isnumeric() == False]
    df_1['terms'] = df_1['source'] + '_' + df_1['target']
    print(df_1.columns)
    # df = df.drop(columns = ['source', 'target'])
    return df_1

def get_files(input_file, output_path):

    input_df = pd.read_csv(input_file)
    types = ['LOC', 'GPE']
    frames = []
    ner_data = IncrementalBar('ner data for volume', max=len(input_df.index))
    for index, row in input_df.iterrows():
        ner_data.next()
        df = pd.DataFrame(input_df.iloc[index]).transpose()
        df.reset_index(inplace=True)
        page_number = df.at[0, 'page_number']
        file_name = df.at[0, 'file_name']
        vol = df.at[0, 'vol']
        date = df.at[0, 'date']
        data = process_text(df, types)
        if not data.empty:
            data['page_number'] = page_number
            data['file_name'] = file_name
            data['vol'] = vol
            data['date'] = date
            frames.append(data)
        else:
            continue
    ner_data.finish()   
    df = pd.concat(frames)
    
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_path, header=True, index=False)
    

if __name__ ==  "__main__" :
	# print(sys.argv[1])
	get_files('1960_1961_arab_observer_image_lucida_final.csv', 'unigram_arab_observer_imagelucida1960_1961.csv')
    # get_files(sys.argv[1], sys.argv[2])

