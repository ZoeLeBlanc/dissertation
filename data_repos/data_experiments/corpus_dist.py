# import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_selection import chi2
from sklearn.decomposition import TruncatedSVD, PCA, NMF, LatentDirichletAllocation
from sklearn.manifold import MDS
import glob,os

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
import random
import codecs, difflib, distance

import warnings
warnings.filterwarnings('ignore')
from itertools import combinations 

def process_page(all_documents, order_text, unorder_text, order_list, unorder_list, ocr_values, page_ocr):
    # Count n grams frequencies and calculate cosine similarity between two docs. 
    counts = CountVectorizer(ngram_range=(1,5))
    counts_matrix = counts.fit_transform(all_documents)
    cos = cosine_similarity(counts_matrix[0:1], counts_matrix)
#     print('Count Vectorizer', cos[0][1])
    ocr_values.append(cos[0][1])
    
    # Calculate tf-idf cosine similarity (nltk or spacy text the same)
    tokenize = lambda doc: doc.lower().split(" ")
    tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize, ngram_range=(1,5))
    tfidf_matrix = tfidf.fit_transform(all_documents)

    cos = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
#     print('TF-IDF Vectorizer', cos[0][1])
    ocr_values.append(cos[0][1])
    
#     # Calculate similarity using GLOVE and SPACY
#     order_doc = nlp(order_text)
#     unorder_doc = nlp(unorder_text)
#     sim_doc = order_doc.similarity(unorder_doc)
# #     print('Spacy GLOVE', sim_doc)
#     #https://stats.stackexchange.com/questions/304217/how-is-the-similarity-method-in-spacy-computed
#     ocr_values.append(sim_doc)
    
    # Calculate jaccard ratio. Takes list of tokens
    jac = 1 - distance.jaccard(order_list, unorder_list)
#     print('Jaccard', jac)
    ocr_values.append(jac)
    
    # use gensim's similarity matrix and lsi to calculate cosine
    all_tokens = [order_list, unorder_list]
    dictionary = Dictionary(all_tokens)
    corpus = [dictionary.doc2bow(text) for text in all_tokens]
    lsi = LsiModel(corpus, id2word=dictionary, num_topics=2)
    sim = MatrixSimilarity(lsi[corpus])
    lsi_cos = [ t[1][1] for t in list(enumerate(sim))]
    lsi_cos = lsi_cos[0]
#     print('LSI', lsi_cos)
    ocr_values.append(lsi_cos)
    #https://radimrehurek.com/gensim/tut3.html
    
#     align = align_pages(order_text, unorder_text)
# #     print('smw', align)
#     ocr_values.append(align)
#     print(ocr_values)
    if os.path.isfile(page_ocr):
        final_metrics = pd.read_csv(page_ocr)
        ocr_values.append(datetime.date.today())
        final_metrics.loc[len(final_metrics.index)] = ocr_values
        final_metrics.to_csv(page_ocr, index=False)
    else:
        ocr_values.append(datetime.date.today())
        cols = ['first_issue_date', 'first_page_number', 'second_issue_date', 'second_page_number','countsvec_cos', 'tfidfvec_cos', 'jaccard_sim', 'lsi_cos', 'date_run']
        final_df = pd.DataFrame([ocr_values], columns=cols)
        final_df.to_csv(page_ocr, index=False)

df_1 = pd.read_csv('combined_pages_date_congo_text_ner_binned.csv')
from ast import literal_eval
import itertools
df = df_1

def get_texts(rows):
    
    x = literal_eval(rows.values[0])
    t = ' '.join(x)
#     model_texts.append(t)
    return t

df_grouped = df.groupby(['date', 'binned', 'page_str'])['token_texts'].apply(get_texts).reset_index()


def get_lists(rows):

    x = literal_eval(rows.values[0])
#     print(len(x))
    combined = list(itertools.chain.from_iterable(x))

    return combined

df_grouped1 = df.groupby(['date', 'binned', 'page_str'])['token_lists'].apply(get_lists).reset_index()
df_all = pd.merge(df_grouped, df_grouped1, on=['date', 'binned', 'page_str'])
df_all['datetime'] = pd.to_datetime(df_all['date'], format='%Y-%B-%d', errors='coerce')
df_all.sort_values(by=['datetime'], inplace=True)
df = df_all
dates = df.date.unique()
t = list(combinations(dates, 2))
for item in t:
    print(item[0], item[1])
    issue_1 = df.loc[df.date == item[0]]
    issue_2 = df.loc[df.date == item[1]]

    #ocr values, first page, date, second_page, 
    ocr_values = [item[0], issue_1.page_str.values[0], item[1], issue_2.page_str.values[0]]
    text_1 = issue_1.token_texts.str.join(sep='')
    text_2 = issue_2.token_texts.str.join(sep='')
    all_documents = [text_1.tolist()[0], text_2.tolist()[0]]
    list_1 = text_1.str.split(' ')
    list_2 = text_2.str.split(' ')
#         print(ocr_values)
    process_page(all_documents, text_1.tolist()[0], text_2.tolist()[0], list_1.tolist()[0], list_2.tolist()[0], ocr_values, 'all_issue_similarity_congo_terms_dates_binned_fixed.csv')
