''' 
First split corpus into containing congo terms or not
then tfidvectorize and classify texts


Then split corpus into two congo groups
then tfidvectorize and classify texts
'''
# import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import MDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.utils import shuffle
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
from progress.bar import IncrementalBar
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

# %load_ext rpy2.ipython
# nlp = spacy.load('en_core_web_lg')

# %load_ext rpy2.ipython
# df = pd.read_csv('../scripts/combined_pages_date_congo_text_ner_binned.csv')
# terms = ['patrice','congo','lumumba','tshombe','leopoldville','belgian','mobutu','kasavubu','katanga']
# input_df = pd.read_csv('../data/combined_all_data_1960_1966_binned.csv')


# input_df['classify'] = ''
# input_df.classify[input_df.tokenized_text.str.contains('|'.join(terms)) == True] = 'congo'
# input_df.classify[input_df.tokenized_text.str.contains('|'.join(terms)) == False] = 'not_congo'

# input_df.to_csv('../data/combined_all_data_1960_1966_binned_congo_classified.csv')
# df = pd.read_csv('../data/combined_all_data_1960_1966_binned_congo_classified.csv')
# # df = df[0:100]
# ner_data = IncrementalBar('ner data for volume', max=len(df.index))

# def custom_tokenize(text):
#     if not text:
# #       print('The text to be tokenized is a None type. Defaulting to blank string.')
#         text = ''
#     return nltk.sent_tokenize(text)
# def clean_texts(rows):
#     ner_data.next()
#     texts = rows.astype(str).tolist()
#     final_doc = []
#     for t in texts: 
#         tokens = custom_tokenize(t)
#         page_terms = ''
#         for t1 in tokens:
            
#             toks = nltk.word_tokenize(t1)
#             for t in toks:
#                 if t in string.punctuation:
#                     continue
#                 elif t in stopwords.words('english'):
#                     continue
                    
#                 elif t.isdigit():
#                     continue
#                 else:
#                     page_terms += t.lower() + ' '
#         final_doc.append(page_terms)
#     print(len(final_doc))
#     return final_doc[0]


# df_grouped = df.groupby(['date','binned', 'page_number'])['tokenized_text'].apply(clean_texts).reset_index()
# ner_data.finish()
# # print(df_grouped.tokenized_text.values)
# df_1 = df.drop(['tokenized_text'], axis=1)

# docs = pd.merge(df_1, df_grouped, on=['date','binned', 'page_number'])
# docs.to_csv('../data/combined_all_data_1960_1966_binned_congo_classified_clean.csv')
docs_big = pd.read_csv('../data/combined_all_data_1960_1966_binned_congo_classified_clean.csv')
docs_big.tokenized_text = docs_big.tokenized_text.fillna('')
docs_big.classify = docs_big.classify.fillna('not_congo')
# print(len(docs[docs.classify =='congo']), len(docs[docs.classify =='not_congo']))
docs = pd.read_csv('../notebooks/training_data.csv')

print(len(docs[docs.classify =='congo']), len(docs[docs.classify =='not_congo']))

# docs.classify[docs.classify =='congo'] = '1'
# docs.classify[docs.classify =='not_congo'] = '0'
# docs.classify = docs.classify.astype(int)
# docs = docs[0:10]
docs.tokenized_text = docs.tokenized_text.fillna('')
df = shuffle(docs)
y = df['classify']
tfidf_model = TfidfVectorizer(lowercase=False)
features = tfidf_model.fit_transform(docs.tokenized_text.tolist())

features_nd = features.toarray()
print(len(features_nd))
training_features, test_features, training_target, test_target = train_test_split(features_nd[0:len(docs['tokenized_text'])], y,test_size=0.8, random_state=53)

print(training_features.shape, test_features.shape, training_target.shape, test_target.shape)

x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,test_size = .1,random_state=12)

sm = SMOTE(random_state=12, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)

log_model = LogisticRegression()
log_model = log_model.fit(X=x_train_res, y=y_train_res)
y_pred = log_model.predict(x_val)
print('Validation Results')
print(log_model.score(x_val, y_val))
print(metrics.recall_score(y_val, y_pred, pos_label='congo'))
print("Precision:",metrics.precision_score(y_val, y_pred, pos_label='congo'))
print('\nTest Results')
print(log_model.score(test_features, test_target))
print(metrics.recall_score(test_target, log_model.predict(test_features), pos_label='congo'))
print("Precision:",metrics.precision_score(test_target, log_model.predict(test_features),pos_label='congo'))

print(len(log_model.predict_proba(test_features)[:, 1]))
# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_model.score(x_val, y_val)))
# # print("Precision:",metrics.precision_score(y_test, y_pred))
# print("Recall:",metrics.recall_score(y_test, y_pred))
from sklearn import model_selection
kfold = model_selection.KFold(n_splits=10, random_state=7)
# scoring = 'accuracy'
results = model_selection.cross_val_score(log_model, x_train_res, y_train_res, cv=kfold)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

# tfidf_model2 = TfidfVectorizer(lowercase=False)
features_big = tfidf_model.transform(docs_big.tokenized_text.tolist())

features_nd_big = features_big.toarray()
print(len(features_nd_big))

predictions_proba = log_model.predict_proba(features_nd_big[0:len(docs_big['tokenized_text'])])
predictions_one = [x[1] for x in predictions_proba]
predictions_zero = [x[0] for x in predictions_proba]
predictions_big = log_model.predict(features_nd_big[0:len(docs_big['tokenized_text'])])


docs_big['prediction_proba_0'] = predictions_zero
docs_big['prediction_proba_1'] = predictions_one
docs_big['prediction'] = predictions_big
docs_big.to_csv('all_data_predictions_proba.csv')
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# accuracy = metrics.r2_score(y_train, results)
# print('Cross-Predicted Accuracy:', accuracy)

# from sklearn.metrics import confusion_matrix
# confusion_matrix = confusion_matrix(y_test, y_pred)
# print('confusion_matrix', confusion_matrix)

# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))

# y_pred_proba = log_model.predict_proba(X_test)[::,1]
# print(y_pred_proba)
# fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
# auc = metrics.roc_auc_score(y_test, y_pred_proba)
# plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
# plt.legend(loc=4)
# plt.show()
# # # plt.scatter(y_test, results)
# # # plt.xlabel('True Values')
# # # plt.ylabel('Predictions')
# # # plt.show()