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

def custom_tokenize(text):
    if not text:
#       print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    return nltk.sent_tokenize(text)
def join_token_lists(rows):

    texts = rows
    final_doc = []
    sentences = []
    for line in texts:
        sentences.extend(nltk.sent_tokenize(line))
    for sentence in sentences:
        sent_tokens = [token.lower() for token in nltk.word_tokenize(sentence) if token not in string.punctuation and token not in stopwords.words('english') and token.isdigit() == False]


        final_doc.append(sent_tokens)
    return final_doc

def join_token_terms(rows):

    texts = rows.astype(str).tolist()
    final_doc = []
    for t in texts: 
        pages = []
        tokens = custom_tokenize(t)
        for t1 in tokens:
            page_terms = ''
            toks = nltk.word_tokenize(t1)
            for t in toks:
                if t in string.punctuation:
                    continue
                elif t in stopwords.words('english'):
                    continue
                    
                elif t.isdigit():
                    continue
                else:
                    page_terms += t.lower() + ' '
            pages.append(page_terms)
        final_doc.append(pages)
    print(len(final_doc))
    return final_doc
def get_texts(rows):

    texts = rows.tolist()
    return texts[0]

df = pd.read_csv('../data/combined_all_data_ner_congo.csv')
d = df
df_tokenized = d.groupby(['date', 'page_number'])['tokenized_text'].apply(get_texts).reset_index()
df_lists = df_tokenized.groupby(['date'])['tokenized_text'].apply(join_token_lists).reset_index()

df_tokens = df_tokenized.groupby(['date'])['tokenized_text'].apply(join_token_terms).reset_index()
df_pages = df_tokenized.groupby(['date'])['page_number'].count().reset_index()
df_tokenized['page_str'] = df_tokenized.page_number.astype(str)
df_exact_pages = df_tokenized.groupby(['date'])['page_str'].apply(' '.join).reset_index()
df_terms = d.groupby(['date'])['term'].apply(' '.join).reset_index()
df_counts = d.groupby(['date'])['word_counts'].sum().reset_index()

dl = pd.merge(df_lists, df_terms, on=['date'])
dx = pd.merge(dl, df_tokens, on=['date'])
dn = pd.merge(dx, df_pages, on=['date'])
db = pd.merge(dn, df_exact_pages, on=['date'])
final_df = pd.merge(db, df_counts, on=['date'])
final_df.rename(columns={'tokenized_text_x': 'token_lists', 'tokenized_text_y': 'token_texts'}, inplace=True)
final_df.to_csv('combined_pages_date_congo_text_ner.csv')
# for i, row in final_df.iterrows():
#     count_df = tfidf_vec(row, (1,2))

# def join_token_terms_types(rows):
#     texts = rows.astype(str).tolist()
#     final_doc = []
#     types = ['LOC', 'GPE']
#     for t in texts: 
#         spacy_terms = ''
#         spacy_text = nlp(t)
#         for ent in spacy_text.ents:
#             print(ent.text)
#             if ent.text in string.punctuation:
#                 continue
#             elif ent.text in stopwords.words('english'):
#                 continue
#             # if ent.label_ in types:
#             elif any(i.isdigit() for i in ent.text) == False:
#                 if '.' in ent.text:
#                     text = ('').join(ent.text.split('.'))
#                     spacy_terms += text + ' '
#                 else:
#                     spacy_terms += ent.text + ' '
#             print(spacy_terms)
#         final_doc.append(spacy_terms)
#     return final_doc


# def join_token_term(rows):
#     texts = rows.astype(str).tolist()
#     final_doc = []
#     for t in texts: 
#         spacy_terms = ''
#         spacy_text = nlp(t)
#         for ent in spacy_text.ents:
#             print(ent.text)
#             if ent.text in string.punctuation:
#                 continue
#             elif ent.text in stopwords.words('english'):
#                 continue
#             # if ent.label_ in types:
#             elif any(i.isdigit() for i in ent.text) == False:
#                 if '.' in ent.text:
#                     text = ('').join(ent.text.split('.'))
#                     spacy_terms += text + ' '
#                 else:
#                     spacy_terms += ent.text + ' '
#             print(spacy_terms)
#         final_doc.append(spacy_terms)
#     return final_doc

# def join_spacy_terms(rows):
#     texts = rows.astype(str).tolist()
#     final_doc = []
#     for t in texts: 
#         spacy_terms = ''
#         spacy_text = nlp(t)
#         for ent in spacy_text.ents:
#             if any(i.isdigit() for i in ent.text) == False:
#                 if '.' in ent.text:
#                     text = ('').join(ent.text.split('.'))
#                     spacy_terms += text + ' '
#                 else:
#                     spacy_terms += ent.text + ' '
#         final_doc.append(spacy_terms)
#     return final_doc

# def build_matrix(input_file, output_file, types, nrange):
#     df = pd.read_csv(input_file)
#     d = df[0:5]
#     df_spacy = d.groupby(['date'])['spacy_text'].apply(join_spacy_terms).reset_index()
#     df_tokenized = d.groupby(['date'])['tokenized_text'].apply(join_token_terms).reset_index()
#     df_pages = d.groupby(['date'])['page_number'].count().reset_index()
#     df_terms = d.groupby(['date'])['term'].apply(' '.join).reset_index()
#     df_counts = d.groupby(['date'])['word_counts'].sum().reset_index()
#     dm = pd.merge(df_spacy, df_tokenized, on=['date'])
#     dl = pd.merge(dm, df_terms, on=['date'])
#     dn = pd.merge(dl, df_pages, on=['date'])
#     final_df = pd.merge(dn, df_counts, on=['date'])
#     # print(final_df)
#     for i, row in final_df.iterrows():
#     #     print(row)

#         count_df = count_vec(row, nrange)
#     #     tfidf_df = tfidf_vec(row, nrange)
#     # # print(tfidf_df)
#     # if os.path.exists(output_file):
#     #     df.to_csv(output_file, mode='a', header=False, index=False)
#     # else:
#     #     df.to_csv(output_file, header=True, index=False)
# #     class MyEncoder(json.JSONEncoder):
# #     def default(self, obj):
# #         if isinstance(obj, np.integer):
# #             return int(obj)
# #         elif isinstance(obj, np.floating):
# #             return float(obj)
# #         elif isinstance(obj, np.ndarray):
# #             return obj.tolist()
# #         else:
# #             return super(MyEncoder, self).default(obj)
# # order_types = ['LOC', 'GPE']
# # jdata = process_text(htrc_df[0:10], order_types)
# # with open('htrc_data.json', 'w') as outfile:
# #         json.dump(jdata, outfile, cls=MyEncoder)



# # def count_vec(docs, nrange):
# #     print(docs)
# #     count_model = CountVectorizer(ngram_range=(nrange)) # default unigram model
# #     X = count_model.fit_transform(docs.tokenized_text)
# #     Xc = (X.T * X)
# #     # Xc.setdiag(0)
# #     lsa = TruncatedSVD(n_components=X.shape[1]-1, n_iter=10)
# #     lsaOut = lsa.fit(X)
# #     xs, ys = lsaOut.components_[0], lsaOut.components_[1]
# #     vocab = count_model.vocabulary_
# #     labels = [x.strip() for x,y in vocab.items()]
# #     feature_names = count_model.get_feature_names()
# #     best_features = [feature_names[i] for i in lsaOut.components_[0].argsort()[::-1]]
# #     print(labels)
# #     for i in range(len(xs)):
# #         # print(labels[i], xs[i], ys[i])
# #         plt.scatter(xs[i], ys[i])
# #         plt.annotate(best_features[i], (xs[i], ys[i]))
# #     plt.show()
# #     # print(pd.DataFrame(X.toarray(), columns=count_model.get_feature_names()[0:100]))
# #     feature_array = np.array(count_model.get_feature_names())
# #     tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]
    
# #     # best_features = [feature_names[i] for i in lsaOut.components_[0].argsort()[::-1]]
# #     print(best_features[:100])
# #     n = 100
# #     top_n = feature_array[tfidf_sorting][:n]
# #     print(top_n)
# #     # print(count_model.get_feature_names())
    
# #     # print(vocab)
# #     vocab2 = {y:x for x,y in vocab.items()}
# #     return create_network(Xc, vocab2, docs)

# # def tfidf_vec(docs, nrange): 
# #     tfidf_model = TfidfVectorizer(ngram_range=(nrange)) # default unigram model
# #     X = tfidf_model.fit_transform(docs.tokenized_text)
# #     Xc = (X.T * X)
# #     # print(tfidf_model.get_feature_names())
# #     print(pd.DataFrame(X.toarray(), columns=tfidf_model.get_feature_names()[0:100]))
# #     # Xc.setdiag(0)
# #     feature_array = np.array(tfidf_model.get_feature_names())
# #     tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]

# #     n = 10
# #     top_n = feature_array[tfidf_sorting][:n]
# #     print(top_n)
# #     vocab = tfidf_model.vocabulary_
# #     vocab2 = {y:x for x,y in vocab.items()}
# #     return create_network(Xc, vocab2, docs)

# # def create_network(matrix, vocab, d):
# #     # print(d)
# #     G = nx.from_scipy_sparse_matrix(matrix)
# #     H = nx.relabel_nodes(G, vocab)
# #     df = nx.to_pandas_edgelist(H)
# #     df = df[df['source'].str.isnumeric() == False]
# #     df = df[df['target'].str.isnumeric() == False]
# #     df['page_number'] = d.page_number
# #     df['date'] = d.date
# #     df['terms'] = d.term
# #     df['word_counts'] = d.word_counts
# #     # df['terms'] = df['source'] + '_' + df['target']
# #     # df = df.loc[(df['weight'] > 0) == True]
# #     print(len(df))
# #     # df = df.drop(columns = ['source', 'target'])
# #     return df




# if __name__ ==  "__main__" :
#     # print(sys.argv[1])
#     types = []
#     ngram_range = (1,1)
#     build_matrix('../data/combined_all_data_ner_congo.csv', '../data/combined_all_data_ner_bigrams.csv', types, ngram_range)
#     # get_files(sys.argv[1], sys.argv[2])

