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

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

# %load_ext rpy2.ipython
# nlp = spacy.load('en_core_web_lg')
import logging
from ast import literal_eval
import itertools
from tmtoolkit.topicmod import tm_gensim
from tmtoolkit.corpus import Corpus
from tmtoolkit.preprocess import TMPreproc
from tmtoolkit.utils import pickle_data
from tmtoolkit.topicmod.evaluate import results_by_parameter
from tmtoolkit.topicmod.visualize import plot_eval_results


logging.basicConfig(level=logging.INFO)
tmtoolkit_log = logging.getLogger('tmtoolkit')
tmtoolkit_log.setLevel(logging.INFO)
tmtoolkit_log.propagate = True
# %load_ext rpy2.ipython
# df = pd.read_csv('../scripts/combined_pages_date_congo_text_ner_binned.csv')
df = pd.read_csv('../scripts/combined_pages_date_binned.csv')
print(len(df))
model_texts = []
def get_texts(rows):
    
    x = literal_eval(rows.values[0])
    t = ' '.join(x)
    model_texts.append(t)
    return t

df_grouped = df.groupby(['date', 'binned'])['token_texts'].apply(get_texts).reset_index()

model_lists = []
def get_lists(rows):

    x = literal_eval(rows.values[0])
#     print(len(x))
    combined = list(itertools.chain.from_iterable(x))

#     print(len(combined))
    model_lists.append(combined)
#     print(len(model_lists))
    return combined

df_grouped1 = df.groupby(['date', 'binned'])['token_lists'].apply(get_lists).reset_index()
print(len(model_texts))

gnsm_dict = Dictionary(model_lists)
# gnsm_dict.filter_extremes(no_above=0.2)
gnsm_corpus = [gnsm_dict.doc2bow(text) for text in model_lists]




# evaluate topic models with different parameters
const_params = dict(update_every=0, passes=20, iterations=400, alpha='auto', eta='auto',)
ks = list(range(10, 140, 10)) + list(range(140, 200, 20))
varying_params = [dict(num_topics=k, alpha=1.0 / k) for k in ks]

print('evaluating %d topic models' % len(varying_params))
eval_results = tm_gensim.evaluate_topic_models((gnsm_dict, gnsm_corpus), varying_params, const_params,coherence_gensim_texts=model_lists)   # necessary for coherence C_V metric

# save the results as pickle
print('saving results')
pickle_data(eval_results, 'gensim_evaluation_results_entire.pickle')

# plot the results
print('plotting evaluation results')
plt.style.use('ggplot')
results_by_n_topics = results_by_parameter(eval_results, 'num_topics')
plot_eval_results(results_by_n_topics, xaxislabel='num. topics k',
                  title='Evaluation results', figsize=(8, 6))
plt.savefig('gensim_evaluation_plot_entire.png')
plt.show()