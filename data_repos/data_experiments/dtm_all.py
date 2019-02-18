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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# %load_ext rpy2.ipython
# nlp = spacy.load('en_core_web_lg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from ast import literal_eval
# %load_ext rpy2.ipython
df = pd.read_csv('all_years_data_tm.csv')
df_1960 = df.loc[df.year == 1960 ]
df_1961 = df.loc[df.year == 1961 ]
df_1962 = df.loc[df.year == 1962 ]
df_1963 = df.loc[df.year == 1963 ]
df_1964 = df.loc[df.year == 1964 ]
df_1965 = df.loc[df.year == 1965 ]
df_1966 = df.loc[df.year == 1966 ]

tfidf_model_zero = TfidfVectorizer(min_df=0.05, max_df=0.30, lowercase=False)
zero = tfidf_model_zero.fit_transform(literal_eval(df_1960.token_lists.values[0]))
tfidf_model_one = TfidfVectorizer(min_df=0.05, max_df=0.30)
one = tfidf_model_one.fit_transform(literal_eval(df_1961.token_texts.values[0]))
tfidf_model_two = TfidfVectorizer(min_df=0.05, max_df=0.30)
two = tfidf_model_two.fit_transform(literal_eval(df_1962.token_texts.values[0]))
tfidf_model_three = TfidfVectorizer(min_df=0.05, max_df=0.30)
three = tfidf_model_three.fit_transform(literal_eval(df_1963.token_texts.values[0]))
tfidf_model_four = TfidfVectorizer(min_df=0.05, max_df=0.30)
four = tfidf_model_four.fit_transform(literal_eval(df_1964.token_texts.values[0]))
tfidf_model_five = TfidfVectorizer(min_df=0.05, max_df=0.30)
five = tfidf_model_five.fit_transform(literal_eval(df_1965.token_texts.values[0]))
tfidf_model_six = TfidfVectorizer(min_df=0.05, max_df=0.30)
six = tfidf_model_six.fit_transform(literal_eval(df_1966.token_texts.values[0]))

corpus_zero = gensim.matutils.Sparse2Corpus(zero, documents_columns=False)
dictionary_zero = Dictionary.from_corpus(corpus_zero,id2word=dict((id, word) for word, id in tfidf_model_zero.vocabulary_.items()))
model_zero = LdaModel(corpus=corpus_zero, id2word=dictionary_zero, iterations=50, num_topics=40)

corpus_one = gensim.matutils.Sparse2Corpus(one, documents_columns=False)
dictionary_one = Dictionary.from_corpus(corpus_one,id2word=dict((id, word) for word, id in tfidf_model_one.vocabulary_.items()))
model_one = LdaModel(corpus=corpus_one, id2word=dictionary_one, iterations=50, num_topics=40)

corpus_two = gensim.matutils.Sparse2Corpus(two, documents_columns=False)
dictionary_two = Dictionary.from_corpus(corpus_two,id2word=dict((id, word) for word, id in tfidf_model_two.vocabulary_.items()))
model_two = LdaModel(corpus=corpus_two, id2word=dictionary_two, iterations=50, num_topics=40)

corpus_three = gensim.matutils.Sparse2Corpus(three, documents_columns=False)
dictionary_three = Dictionary.from_corpus(corpus_three,id2word=dict((id, word) for word, id in tfidf_model_three.vocabulary_.items()))
model_three = LdaModel(corpus=corpus_three, id2word=dictionary_three, iterations=50, num_topics=40)

corpus_four = gensim.matutils.Sparse2Corpus(four, documents_columns=False)
dictionary_four = Dictionary.from_corpus(corpus_four,id2word=dict((id, word) for word, id in tfidf_model_four.vocabulary_.items()))
model_four = LdaModel(corpus=corpus_four, id2word=dictionary_four, iterations=50, num_topics=40)

corpus_five = gensim.matutils.Sparse2Corpus(five, documents_columns=False)
dictionary_five = Dictionary.from_corpus(corpus_five,id2word=dict((id, word) for word, id in tfidf_model_five.vocabulary_.items()))
model_five = LdaModel(corpus=corpus_five, id2word=dictionary_five, iterations=50, num_topics=40)

corpus_six = gensim.matutils.Sparse2Corpus(six, documents_columns=False)
dictionary_six = Dictionary.from_corpus(corpus_six,id2word=dict((id, word) for word, id in tfidf_model_six.vocabulary_.items()))
model_six = LdaModel(corpus=corpus_six, id2word=dictionary_six, iterations=50, num_topics=40)

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
# limit=150; start=2; step=5;
# x = range(start, limit, step)
# # model_list, coherence_values = compute_coherence_values(dictionary=dictionary_one, corpus=corpus_one, texts=one, start=2, limit=150, step=5)
# # # # Show graph

# # plt.plot(x, coherence_values)
# # plt.xlabel("Num Topics")
# # plt.ylabel("Coherence score")
# # plt.legend(("coherence_values"), loc='best')
# # plt.show()

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

# cm_mass_zero = CoherenceModel(model=model_zero, corpus=corpus_zero, dictionary=dictionary_zero, coherence='u_mass')
# cm_mass_one = CoherenceModel(model=model_one, corpus=corpus_one, dictionary=dictionary_one, coherence='u_mass')
# cm_mass_two = CoherenceModel(model=model_two, corpus=corpus_two, dictionary=dictionary_two, coherence='u_mass')
# cm_mass_three = CoherenceModel(model=model_three, corpus=corpus_three, dictionary=dictionary_three, coherence='u_mass')
# cm_mass_four = CoherenceModel(model=model_four, corpus=corpus_four, dictionary=dictionary_four, coherence='u_mass')
# cm_mass_five = CoherenceModel(model=model_five, corpus=corpus_five, dictionary=dictionary_five, coherence='u_mass')
# cm_mass_six = CoherenceModel(model=model_six, corpus=corpus_six, dictionary=dictionary_six, coherence='u_mass')

# print('zero mass', cm_mass_zero.get_coherence(),'one mass', cm_mass_one.get_coherence(),'two mass', cm_mass_two.get_coherence(),'three mass', cm_mass_three.get_coherence(),'four mass', cm_mass_four.get_coherence(), 'five mass', cm_mass_five.get_coherence(), 'six mass', cm_mass_six.get_coherence())

# cm_cv_zero = CoherenceModel(model=model_zero, texts=zero, dictionary=dictionary_zero, coherence='c_v')
# cm_cv_one = CoherenceModel(model=model_one, texts=one, dictionary=dictionary_one, coherence='c_v')
# cm_cv_two = CoherenceModel(model=model_two, texts=two, dictionary=dictionary_two, coherence='c_v')
# cm_cv_three = CoherenceModel(model=model_three, texts=three, dictionary=dictionary_three, coherence='c_v')
# cm_cv_four = CoherenceModel(model=model_four, texts=four, dictionary=dictionary_four, coherence='c_v')
# cm_cv_five = CoherenceModel(model=model_five, texts=five, dictionary=dictionary_five, coherence='c_v')
# cm_cv_six = CoherenceModel(model=model_six, texts=six, dictionary=dictionary_six, coherence='c_v')

# print('zero cv', cm_cv_zero.get_coherence(),'one cv', cm_cv_one.get_coherence(),'two cv', cm_cv_two.get_coherence(),'three cv', cm_cv_three.get_coherence(),'four cv', cm_cv_four.get_coherence(), 'five cv', cm_cv_five.get_coherence(), 'six cv', cm_cv_six.get_coherence())


models = [model_zero, model_one, model_two, model_three, model_four, model_five, model_six]
corpus = [corpus_zero, corpus_one, corpus_two, corpus_three, corpus_four, corpus_five, corpus_six]
years = ['1960', '1961', '1962', '1963', '1964', '1965', '1966']
processing = IncrementalBar('processing text', max=len(years))
for i,m in enumerate(models):
    processing.next()
    all_topics = m.get_document_topics(corpus[i], per_word_topics=True)

    output_path = years[i]+'_all_lda_words.csv'
    counter = 0 
    for doc_topics, word_topics, phi_values in all_topics:
        print(counter)
        
        for topic in doc_topics:
            print('top', topic[0])
            for (w, weight) in m.show_topic(topic[0], topn=50):

            
                d = {}
                d['word'] = w
                d['doc_page'] = counter
                d['year'] = years[i]
                d['word_weight'] = weight
                d['topic_id'] = topic[0]
                d['topic_weight'] = topic[1]
                dl = pd.DataFrame().append(d, ignore_index=True)
                if os.path.exists(output_path):
                    dl.to_csv(output_path, mode='a', header=False, index=False)
                else:
                    dl.to_csv(output_path, header=True, index=False)
        counter= counter+1
processing.finish()

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