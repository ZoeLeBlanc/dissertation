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
# from datasketch import MinHash
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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# %load_ext rpy2.ipython
# nlp = spacy.load('en_core_web_lg')
from ast import literal_eval
import itertools
# %load_ext rpy2.ipython
df = pd.read_csv('congo_years_data_tm.csv')
model_texts = []
def get_texts(rows):
    
    x = literal_eval(rows.values[0])
    t = ' '.join(x)
    model_texts.append(t)
    return t

df_grouped = df.groupby(['year'])['token_texts'].apply(get_texts)

model_lists = []
def get_lists(rows):

    x = literal_eval(rows.values[0])
    combined = list(itertools.chain.from_iterable(x))

    model_lists.append(combined)
    return x

print(len(model_texts))
df_grouped1 = df.groupby(['year'])['token_lists'].apply(get_lists)

tfidf_model = TfidfVectorizer(lowercase=False)
tfidf = tfidf_model.fit_transform(model_texts)

corpus = gensim.matutils.Sparse2Corpus(tfidf, documents_columns=False)
dictionary = Dictionary.from_corpus(corpus,id2word=dict((id, word) for word, id in tfidf_model.vocabulary_.items()))
model = LdaModel(corpus=corpus, id2word=dictionary, iterations=50, num_topics=80)
model.save('lda_congo.model')

# def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
#     """
#     Compute c_v coherence for various number of topics

#     Parameters:
#     ----------
#     dictionary : Gensim dictionary
#     corpus : Gensim corpus
#     texts : List of input texts
#     limit : Max num of topics

#     Returns:
#     -------
#     model_list : List of LDA topic models
#     coherence_values : Coherence values corresponding to the LDA model with respective number of topics
#     """
#     coherence_values = []
#     model_list = []
#     for num_topics in range(start, limit, step):
#         model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
#         model_list.append(model)
#         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())

#     return model_list, coherence_values
# limit=80; start=2; step=20;
# x = range(start, limit, step)
# model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=model_lists, start=start, limit=limit, step=step)
# # # # # Show graph

# plt.plot(x, coherence_values)
# plt.xlabel("Num Topics")
# plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
# plt.show()

# model_list2, coherence_values2 = compute_coherence_values(dictionary=dictionary_two, corpus=corpus_two, texts=two, start=2, limit=150, step=5)
# # Show graph
# # import matplotlib.pyplot as plt

# plt.plot(x, coherence_values2)
# plt.xlabel("Num Topics")
# plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
# plt.show()
# print(df.year.unique())

# dm = []
# for i, row in df.iterrows():
#     print(i, len(literal_eval(row.token_lists)))
#     dm = dm + literal_eval(row.token_lists)

# time_seq = [117, 327, 314, 144, 157, 108, 55]
# dictionary = Dictionary(dm)
# corpus = [dictionary.doc2bow(text) for text in dm]
# from gensim.models import ldaseqmodel
# ldaseq = ldaseqmodel.LdaSeqModel(corpus=corpus, id2word=dictionary, time_slice=time_seq, num_topics=5)
# ldaseq.print_topics(time=1)

cm_mass = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
# cm_mass_one = CoherenceModel(model=model_one, corpus=corpus_one, dictionary=dictionary_one, coherence='u_mass')
# cm_mass_two = CoherenceModel(model=model_two, corpus=corpus_two, dictionary=dictionary_two, coherence='u_mass')
# cm_mass_three = CoherenceModel(model=model_three, corpus=corpus_three, dictionary=dictionary_three, coherence='u_mass')
# cm_mass_four = CoherenceModel(model=model_four, corpus=corpus_four, dictionary=dictionary_four, coherence='u_mass')
# cm_mass_five = CoherenceModel(model=model_five, corpus=corpus_five, dictionary=dictionary_five, coherence='u_mass')
# cm_mass_six = CoherenceModel(model=model_six, corpus=corpus_six, dictionary=dictionary_six, coherence='u_mass')
print('mass', cm_mass.get_coherence())
# print('zero mass', cm_mass_zero.get_coherence(),'one mass', cm_mass_one.get_coherence(),'two mass', cm_mass_two.get_coherence(),'three mass', cm_mass_three.get_coherence(),'four mass', cm_mass_four.get_coherence(), 'five mass', cm_mass_five.get_coherence(), 'six mass', cm_mass_six.get_coherence())

cm_cv = CoherenceModel(model=model, texts=model_texts, dictionary=dictionary, coherence='c_v')
print('cv', cm_cv.get_coherence())
# cm_cv_one = CoherenceModel(model=model_one, texts=one, dictionary=dictionary_one, coherence='c_v')
# cm_cv_two = CoherenceModel(model=model_two, texts=two, dictionary=dictionary_two, coherence='c_v')
# cm_cv_three = CoherenceModel(model=model_three, texts=three, dictionary=dictionary_three, coherence='c_v')
# cm_cv_four = CoherenceModel(model=model_four, texts=four, dictionary=dictionary_four, coherence='c_v')
# cm_cv_five = CoherenceModel(model=model_five, texts=five, dictionary=dictionary_five, coherence='c_v')
# cm_cv_six = CoherenceModel(model=model_six, texts=six, dictionary=dictionary_six, coherence='c_v')

# print('zero cv', cm_cv_zero.get_coherence(),'one cv', cm_cv_one.get_coherence(),'two cv', cm_cv_two.get_coherence(),'three cv', cm_cv_three.get_coherence(),'four cv', cm_cv_four.get_coherence(), 'five cv', cm_cv_five.get_coherence(), 'six cv', cm_cv_six.get_coherence())


# models = [model_zero, model_one, model_two, model_three, model_four, model_five, model_six]
# corpus = [corpus_zero, corpus_one, corpus_two, corpus_three, corpus_four, corpus_five, corpus_six]
# years = ['1960', '1961', '1962', '1963', '1964', '1965', '1966']
# processing = IncrementalBar('processing text', max=len(years))
# for i,m in enumerate(models):
#     processing.next()
#     all_topics = m.get_document_topics(corpus[i], per_word_topics=True)

#     output_path = years[i]+'_all_lda_words.csv'
#     counter = 0 
#     for doc_topics, word_topics, phi_values in all_topics:
#         print(counter)
        
#         for topic in doc_topics:
#             print('top', topic[0])
#             for (w, weight) in m.show_topic(topic[0], topn=50):

            
#                 d = {}
#                 d['word'] = w
#                 d['doc_page'] = counter
#                 d['year'] = years[i]
#                 d['word_weight'] = weight
#                 d['topic_id'] = topic[0]
#                 d['topic_weight'] = topic[1]
#                 dl = pd.DataFrame().append(d, ignore_index=True)
#                 if os.path.exists(output_path):
#                     dl.to_csv(output_path, mode='a', header=False, index=False)
#                 else:
#                     dl.to_csv(output_path, header=True, index=False)
#         counter= counter+1
# processing.finish()

# lda_1960 = pd.read_csv('../scripts/1960_all_lda_words.csv')
# lda_1961 = pd.read_csv('../scripts/1961_all_lda_words.csv')
# lda_1962 = pd.read_csv('../scripts/1962_all_lda_words.csv')
# lda_1963 = pd.read_csv('../scripts/1963_all_lda_words.csv')
# lda_1964 = pd.read_csv('../scripts/1964_all_lda_words.csv')
# lda_1965 = pd.read_csv('../scripts/1965_all_lda_words.csv')
# lda_1966 = pd.read_csv('../scripts/1966_all_lda_words.csv')

# print('1960',list(lda_1960.word.unique()))
# print('1961',list(lda_1961.word.unique()))
# print('1962',list(lda_1962.word.unique()))
# print('1963',list(lda_1963.word.unique()))
# print('1964',list(lda_1964.word.unique()))
# print('1965',list(lda_1965.word.unique()))
# print('1966',list(lda_1966.word.unique()))

# words_1960 =list(lda_1960.word.unique())
# words_1961 =list(lda_1961.word.unique())
# words_1962 =list(lda_1962.word.unique())
# words_1963 =list(lda_1963.word.unique())
# words_1964 =list(lda_1964.word.unique())
# words_1965 =list(lda_1965.word.unique())
# words_1966 =list(lda_1966.word.unique())

# words = [
#     words_1960,
#     words_1961,
#     words_1962,
#     words_1963,
#     words_1964,
#     words_1965,
#     words_1966,
# ]
# for i, w in enumerate(words):
#     if i+1< len(words):
#         print(len(w), len(words[i+1]), len(list(set(w)-set(words[i+1]))))

# df_tokenized = d.groupby(['year'])['tokenized_text'].apply(get_texts).reset_index()
# df_lists = df_tokenized.groupby(['date'])['tokenized_text'].apply(join_token_lists).reset_index()

# df_tokens = df_tokenized.groupby(['date'])['tokenized_text'].apply(join_token_terms).reset_index()
# df_pages = df_tokenized.groupby(['date'])['page_number'].count().reset_index()
# df_terms = d.groupby(['date'])['term'].apply(' '.join).reset_index()
# df_counts = d.groupby(['date'])['word_counts'].sum().reset_index()

# dl = pd.merge(df_lists, df_terms, on=['date'])
# dx = pd.merge(dl, df_tokens, on=['date'])
# dn = pd.merge(dx, df_pages, on=['date'])
# final_df = pd.merge(dn, df_counts, on=['date'])
# final_df.rename(columns={'tokenized_text_x': 'token_lists', 'tokenized_text_y': 'token_texts'}, inplace=True)
# final_df.to_csv('combined_pages_date_congo_text_ner.csv')