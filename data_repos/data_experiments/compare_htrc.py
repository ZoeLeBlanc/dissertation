import pandas as pd 
import string
import os
import glob
import numpy as np 
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import gensim
from gensim.corpora import Dictionary
from gensim.similarities import MatrixSimilarity
from gensim.models import ldamodel, doc2vec, LsiModel 
import nltk
# nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.util import ngrams
# nltk.download('stopwords')
from collections import OrderedDict, Counter, namedtuple
import datetime
import  distance

"""Need to read in the htrc pages, and compare to individual pages of the scribe. 
1. HTRC -> compare one page to one page of scribe
2. HTRC -> join text to compare entirety of texts to scribe
3. scribe -> 
"""

scribe_df = pd.read_csv('scribe_feb_1_1964_unordered.csv')
page_6 = scribe_df.iloc[[5]]
filenames = []
for file in glob.glob('./htrc_pages/*.csv'):
    filenames.append(file)

# for file in filenames:
#     df = pd.read_csv(file)
#     df['lowercase'] = df['lowercase'].astype(str)
#     selected = df[df['lowercase'].astype(str).str.contains('Camille|camille')].copy()
#     if len(selected.index) > 0 :
#         print(selected)
files = []
# for i in range(0, 150):
#     file = [ filename for filename in filenames if str(i) in filename]
#     for f in file:
#         if f.split('/')[2].split('_')[0] == str(i):
#             print(f)
#             files.append(f)
htrc_df = pd.DataFrame()
for i, file in enumerate(filenames):
    if i < 1: 
        htrc_df = pd.read_csv(file)
    else:
        df = pd.read_csv(file)
        frames = [htrc_df, df]
        htrc_df = pd.concat(frames)

htrc_df = htrc_df.drop(columns=['section', 'count'])

htrc_df['lowercase'] = htrc_df['lowercase'].astype(str)
groupby_df = htrc_df.groupby('page')['lowercase'].apply(' '.join).reset_index()
htrc_df = htrc_df.drop_duplicates(subset=['page'], keep='first')
htrc_df = htrc_df.drop(columns='lowercase')
final_df = pd.merge(htrc_df, groupby_df, on='page', how='outer')
htrc_df = final_df[['page', 'lowercase']]
# print(htrc_df)
htrc_df.to_csv('htrc_grouped.csv')
# def custom_tokenize(text):
#     if not text:
# #       print('The text to be tokenized is a None type. Defaulting to blank string.')
#         text = ''
#     return nltk.word_tokenize(text)

# def process_text(df, df_row):

#     final_doc = []
#     pages = []
#     raw_text = df[df_row]
#     # print(type(raw_text))
#     tokens = custom_tokenize(raw_text)
#     for t in tokens:
#         if t.lower() in string.punctuation:
#             continue
#         elif t.lower() in stopwords.words('english'):
#             continue
#         else:
#             final_doc.append(t.lower())
#     text = ' '.join(final_doc)
#     return final_doc, text


# def process_page(all_documents, order_text, unorder_text, order_list, unorder_list, ocr_values, page_ocr):
    
#     # Calculate tf-idf cosine similarity (nltk or spacy text the same)
#     tokenize = lambda doc: doc.lower().split(" ")
#     tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize, ngram_range=(1,1))
#     tfidf_matrix = tfidf.fit_transform(all_documents)

#     cos = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
#     print('TF-IDF Vectorizer', cos)
#     if cos[0][1] > 0.5:
#         ocr_values.append(cos[0][1])
    
    
#     # Calculate jaccard ratio. Takes list of tokens
#     jac = 1 - distance.jaccard(order_list, unorder_list)
#     print('Jaccard', jac)
#     if jac > 0.5:
#         ocr_values.append(jac)
    
#     # use gensim's similarity matrix and lsi to calculate cosine
#     all_tokens = [order_list, unorder_list]
#     dictionary = Dictionary(all_tokens)
#     corpus = [dictionary.doc2bow(text) for text in all_tokens]
#     lsi = LsiModel(corpus, id2word=dictionary, num_topics=2)
#     sim = MatrixSimilarity(lsi[corpus])
#     lsi_cos = [ t[1][1] for t in list(enumerate(sim))]
#     lsi_cos = lsi_cos[0]
#     # print('LSI', lsi_cos)
#     if lsi_cos > 0.5:
#         ocr_values.append(lsi_cos)
#     #https://radimrehurek.com/gensim/tut3.html
#     file_name = './scribe/' + page_ocr
#     ocr_values.append(len(order_text))
#     ocr_values.append(len(unorder_text))
#     ocr_values.append(datetime.date.today())
#     if len(ocr_values) == 8:
#         cols = ['htrc_page', 'scribe_page', 'tfidfvec_cos', 'jaccard_sim', 'lsi_cos','smw_align', 'len_unorder', 'date_run']
#         print(cols, ocr_values)

#         final_df = pd.DataFrame([ocr_values], columns=cols)
#         final_df.to_csv(page_ocr, index=False)


# for i, row in htrc_df.iterrows():
#     htrc_docs, htrc_text = process_text(row, 'lowercase')
#     for index, r in scribe_df.iterrows():
#         ocr_values = [row['page']]
#         scribe_docs, scribe_text = process_text(r, 'google_vision_text')
#         ocr_values.append(r['page_number'])
#         all_documents = [htrc_text, scribe_text]
#         file_name = 'scribe_htrc_'+str(r['page_number'])+'.csv'
#         process_page(all_documents, htrc_text, scribe_text, htrc_docs, scribe_docs, ocr_values, file_name)