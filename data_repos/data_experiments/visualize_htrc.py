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
# import gensim
# from gensim.corpora import Dictionary
# from gensim.similarities import MatrixSimilarity
# from gensim.models import ldamodel, doc2vec, LsiModel 
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
# from collections import OrderedDict, Counter, namedtuple
# import networkx as nx
# import matplotlib.pyplot as plt
# from networkx.readwrite import json_graph
# from bokeh.plotting import figure, show, output_file
# from bokeh.models import HoverTool, ColumnDataSource
# from bokeh.layouts import row, column
from progress.bar import IncrementalBar
import warnings
warnings.filterwarnings('ignore')
# nlp = spacy.load('en_core_web_lg')
# %load_ext rpy2.ipython

# files = [i for i in glob.glob('./Arab_Observer_HTRC/*.{}'.format('csv')) if 'grouped' in i and 'no' not in i]
# files.sort()
# print(files)

# htrc_df = pd.read_csv(files[0])[0:0]

# for filename in files:
#     df = pd.read_csv(filename, index_col=False)
#     file_name = filename.split('/')[2]
#     df['htrc_vol'] = file_name.split('.')[0]
#     df['years'] = df.htrc_vol.str.split('_').str[3]
#     df['months'] = df.htrc_vol.str.split('_').str[4]
#     df['word_count'] = df['lowercase'].str.split().str.len()
#     htrc_df= htrc_df.append(df, sort=False, ignore_index=True)

# htrc_df.to_csv('Arab_Observer_Distributions_Untokenized.csv')

def custom_tokenize(text):
    if not text:
#       print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    return nltk.word_tokenize(text)

def process_text(df):
    df_1 = df[0:0]
    df.reset_index(inplace=True)
    doc = []
    final_doc = []
    ner_data = IncrementalBar('tokenizing words', max=len(df.index))
    for index, row in df.iterrows():
        ner_data.next()
        raw_text = row['lowercase']
        print(type(raw_text))
        tokens = custom_tokenize(raw_text)
        page_terms = ''
        for t in tokens:
            if t in string.punctuation:
                pass
            elif t in stopwords.words('english'):
                pass
            else:
                page_terms += t.lower() + ' '
        row.tokenized = page_terms
        print(row)
        df_1 = df_1.append(row, ignore_index=True, sort=True)
    df_1.token_counts = df_1.tokenized.str.split().str.len()
    ner_data.finish()
    return df_1

htrc_df = pd.read_csv('Arab_Observer_Distributions_Untokenized.csv')
nltk_df = htrc_df
nltk_df['tokenized'] = ''
nltk_df['token_counts'] = 0
nltk_df = process_text(nltk_df)
nltk_df.to_csv('Arab_Observer_Tokenized_Distributions.csv')