
# coding: utf-8

# In[12]:


import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import gensim
from gensim.corpora import Dictionary
from gensim.models import ldamodel
from gensim.matutils import kullback_leibler, jaccard, hellinger, sparse2full
import nltk
# nltk.download('punkt')
import string
import csv
from nltk.corpus import stopwords
# nltk.download('stopwords')
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer

import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.layouts import row, column
import random


import importlib
ldavis = importlib.util.find_spec("pyLDAvis")
import pyLDAvis.gensim as gensimvis
import pyLDAvis

# In[8]:


#LOAD DATA for each case
full_df=pd.read_csv('full_page_image_lucida_test.csv')
split_df = pd.read_csv('split_page_image_lucida_test.csv', encoding='utf-8')
# order_df.isnull().values.any()
order_df = pd.read_csv('ordered_text_image_lucida_test.csv', encoding="ISO-8859-1")
order_df.fillna(0, inplace=True)

#
# # In[9]:
#
#
# nlp = spacy.load('en_core_web_lg')
#
#
# # In[ ]:
#
#
# # Check Co-Occurence Matrices
#
#
# # In[43]:
#
#
# def custom_tokenize(text):
#     if not text:
# #       print('The text to be tokenized is a None type. Defaulting to blank string.')
#         text = ''
#     return nltk.word_tokenize(text)
#
# def process_text(df, types, graph_settings):
#     doc = []
#     final_doc = []
#     for index, row in df.iterrows():
#         raw_text = row['google_vision_text']
#         tokens = custom_tokenize(raw_text)
#         page_terms = ''
#         for t in tokens:
#             if t in string.punctuation:
#                 pass
#             elif t in stopwords.words('english'):
#                 pass
#             else:
#                 page_terms += t.lower() + ' '
#         doc.append(page_terms)
#
#     for sent in doc:
#         sent_terms = ''
#         spacy_text = nlp(sent)
#         for ent in spacy_text.ents:
#             if ent.label_ in types:
#                 sent_terms += ent.text + ' '
#         final_doc.append(sent_terms)
#     return create_matrix(final_doc, graph_settings)
#
# def create_matrix(ents, graph_settings):
#     count_model = CountVectorizer(ngram_range=(1,1)) # default unigram model
#     X = count_model.fit_transform(ents)
#     Xc = (X.T * X)
#     vocab = count_model.vocabulary_
#     vocab2 = {y:x for x,y in vocab.items()}
#     return create_network(Xc, vocab2, graph_settings)
#     #ALTERNATIVE WAY TO COMPUTE MATRIX
#     # occurrences = OrderedDict((name, OrderedDict((name, 0) for name in termSplit)) for name in termSplit)
#     # # Find the co-occurrences:
#     # for l in document:
#     #     for i in range(len(l)):
#     #         for item in l[:i] + l[i + 1:]:
#     #             occurrences[l[i]][item] += 1
#     # # Print the matrix:
#     # print(' ', ' '.join(occurrences.keys()))
#     # for name, values in occurrences.items():
#     #     print(name, ' '.join(str(i) for i in values.values()))
#
# def create_network(matrix, vocab, graph_settings):
#     G = nx.from_scipy_sparse_matrix(matrix)
#     H = nx.relabel_nodes(G, vocab)
#     data = json_graph.node_link_data(H)
#     T = json_graph.node_link_graph(data)
#     ns = list(T.nodes)
#     es = list(T.edges)
#     final_nodes = []
#     for n in G.nodes:
#         nod = {'name': ns[n], 'id':n}
#         final_nodes.append(nod)
#
#     N = len(T.nodes)
#     counts = np.zeros((N, N))
#     for e in G.edges(data=True):
#         source, target, w = e
#         counts[[source], [target]] = w['weight']
#         counts[[target], [source]] = w['weight']
#     print(len(final_nodes))
#     return draw_graph(counts, final_nodes, graph_settings, ns)
#
# def draw_graph(counts, nodes, graph_settings, list_nodes):
#     xname = []
#     yname = []
#     color = []
#     alpha = []
# #     colormap = ["#444444", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99","#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"]
#     for i, node1 in enumerate(nodes):
#         for j, node2 in enumerate(nodes):
#             xname.append(node1['name'])
#             yname.append(node2['name'])
#
#             alpha.append(min(counts[i,j]/4.0, 0.9) + 0.1)
#
#     for i in range(len(xname)):
#         al = alpha[i]
#         if  al == 0.35:
#             color.append('#ce93d8')
#         elif al == 0.6:
#             color.append('#ba68c8')
#         elif al == 0.85:
#             color.append('#9c27b0')
#         elif al == 1.0:
#             color.append('#7b1fa2')
#         else:
#             color.append('lightgrey')
#
#     source = ColumnDataSource(data=dict(
#         xname=xname,
#         yname=yname,
#         colors=color,
#         alphas=alpha,
#         count=counts.flatten(),
#     ))
#
#     p = figure(title=graph_settings['title'],
#                x_axis_location="above", tools="hover,save",
#                x_range=list(reversed(list_nodes)), y_range=list_nodes)
#
#     p.plot_width = graph_settings['width']
#     p.plot_height = graph_settings['height']
#     p.grid.grid_line_color = None
#     p.axis.axis_line_color = None
#     p.axis.major_tick_line_color = None
#     p.axis.major_label_text_font_size = "5pt"
#     p.axis.major_label_standoff = 0
#     p.xaxis.major_label_orientation = np.pi/3
#
#     p.rect('xname', 'yname', 0.9, 0.9, source=source,
#            color='colors', alpha='alphas', line_color=None,
#            hover_line_color='black', hover_color='colors')
#
#     p.select_one(HoverTool).tooltips = [
#         ('names', '@yname, @xname'),
#         ('count', '@count'),
#     ]
#
#
#
#     return p # show the plot
#
#
# # In[44]:
#
#
# output_file("OCR_coocurence_matrix.html", title='Co-Occurence Test')
# order_settings = {'title': 'Ordered_Text_AO', 'height': 600, 'width': 600}
# order_types = ['GPE']
# order_p = process_text(order_df, order_types, order_settings)
#
# full_settings = {'title': 'Full_Text_AO', 'height': 600, 'width': 600}
# full_types = ['GPE']
# full_p = process_text(full_df, full_types, full_settings)
#
# split_settings = {'title': 'Split_Text_AO', 'height': 600, 'width': 600}
# split_types = ['GPE']
# split_p = process_text(split_df, split_types, split_settings)
#
# show(row(order_p, split_p, full_p))
#
#
# # In[ ]:
#
#
# #Check Topics
#
#
# # In[9]:
#

def custom_tokenize(text):
    if not text:
#       print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    return nltk.word_tokenize(text)

def process_model_text(df):

    final_doc = []
    for index, row in df.iterrows():
        raw_text = row['google_vision_text']
        tokens = custom_tokenize(raw_text)
        doc = []
        for t in tokens:

            if t in string.punctuation:
                pass
            elif t in stopwords.words('english'):
                pass
            else:
                doc.append(t.lower())
        final_doc.append(doc)
    create_models(final_doc)

def create_models(texts):
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20, passes=10)
    print(lda.show_topics())
    vis_data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    print(vis_data)
    pyLDAvis.show(vis_data)

# In[10]:

process_model_text(order_df)

#
#
# # In[ ]:
#
#
# # Check Most Similar Words
#
#
# # In[ ]:
#
#
# def custom_tokenize(text):
#     if not text:
# #       print('The text to be tokenized is a None type. Defaulting to blank string.')
#         text = ''
#     return nltk.word_tokenize(text)
#
# def process_text(df):
#     doc = []
#     final_doc = []
#     for index, row in df.iterrows():
#         raw_text = row['google_vision_text']
#         tokens = custom_tokenize(raw_text)
#         page_terms = ''
#         for t in tokens:
#             if t in string.punctuation:
#                 pass
#             elif t in stopwords.words('english'):
#                 pass
#             else:
#                 page_terms += t.lower() + ' '
#         doc.append(page_terms)
#
#     for sent in doc:
#         sent_terms = ''
#         spacy_text = nlp(sent)
#         for ent in spacy_text.ents:
#             if ent.label_ in types:
#                 sent_terms += ent.text + ' '
#         final_doc.append(sent_terms)
#     return create_matrix(final_doc, graph_settings)
#
