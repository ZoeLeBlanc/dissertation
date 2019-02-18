import re
import pandas as pd
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph
import spacy
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
import csv


def create_matrix_from_character_list(text_file_path, character_list):
    num_of_pages = pd.read_csv(text_file_path)
    df_character_page = pd.DataFrame(columns=['page_number', 'character', 'position'])

def create_matrix_from_csv_of_named_entities(text_file_path):
    num_of_pages = pd.read_csv(text_file_path, encoding="ISO-8859-1")
    num_of_pages.fillna(0, inplace=True)
    nlp = spacy.load('en_core_web_lg')
    new_df = num_of_pages.groupby(['page'])['google_vision_text'].apply(','.join).reset_index()
    doc = []
    final_doc = []
    types = ['PERSON', 'GPE', 'NORP', 'ORG']
    for index, row in new_df.iterrows():
        raw_text = row['google_vision_text']
        page_terms = ''
        for token in nltk.word_tokenize(raw_text):
            if token in string.punctuation:
                pass
            elif token in stopwords.words('english'):
                pass
            else:
                page_terms += token.lower() + ' '
        doc.append(page_terms)

    for sent in doc:
        sent_terms = ''
        spacy_text = nlp(sent)
        for ent in spacy_text.ents:
            if ent.label_ in types:
                sent_terms += ent.text + ' '
        final_doc.append(sent_terms)
    count_model = CountVectorizer(ngram_range=(1,1)) # default unigram model
    X = count_model.fit_transform(final_doc)
    # print(X)
    Xc = (X.T * X) # this is co-occurrence matrix in sparse csr format
    #
    Xc.setdiag(0) # sometimes you want to fill same word cooccurence to 0
    # vocab = []
    # Xc.eliminate_zeros()
    # linked = Xc.tolil()
    # keys = Xc.todok()
    # print(type(keys))
    vocab = count_model.vocabulary_
    vocab2 = {y:x for x,y in vocab.items()}
    G = nx.from_scipy_sparse_matrix(Xc)
    H = nx.relabel_nodes(G, vocab2)
    # print(list(H.nodes), list(H.edges(data=True)))
    data = json_graph.node_link_data(H)
    print(data)
    T = json_graph.node_link_graph(data)
    print(T)

create_matrix_from_csv_of_named_entities('ordered_text_image_lucida_test.csv')
