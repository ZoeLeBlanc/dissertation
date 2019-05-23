
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import MDS
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
 
import warnings
warnings.filterwarnings('ignore')

# full_corpus_df = pd.read_json('../data/arab_observer_corpus_cleaned.json', orient='records')
full_corpus_df = pd.read_csv('../data/arab_observer_corpus_cleaned.csv')
full_corpus_df['datetime'] = pd.to_datetime(full_corpus_df['date'], format='%Y-%B-%d', errors='coerce')

full_corpus_df.cleaned_spacy_text = full_corpus_df.cleaned_spacy_text.fillna('')

congo_df = pd.read_csv('../data/congo_classifier_training_test.data.csv')
congo_filename = '../data/congo_classifier'

congo_df['class_numb'] = 0
congo_df.class_numb[congo_df.classify =='early_congo'] = '0'
congo_df.class_numb[congo_df.classify =='late_congo'] = '1'
congo_df.class_numb = congo_df.class_numb.astype(int)
congo_df.cleaned_spacy_text = congo_df.cleaned_spacy_text.fillna('')



def classify_corpus(df, tfidf_model, log_model, file_name):
    # tfidf_model = TfidfVectorizer(lowercase=False)
    
    features_big = tfidf_model.transform(df.cleaned_spacy_text.tolist())
    features_nd_big = features_big.toarray()

    predictions_proba = log_model.predict_proba(features_nd_big[0:len(df['cleaned_spacy_text'])])
    predictions_one = [x[1] for x in predictions_proba]
    predictions_zero = [x[0] for x in predictions_proba]
    predictions_big = log_model.predict(features_nd_big[0:len(df['cleaned_spacy_text'])])


    df['prediction_proba_0'] = predictions_zero
    df['prediction_proba_1'] = predictions_one
    df['prediction'] = predictions_big
    df.to_csv(file_name + '_classified_corpus.csv')


tfidf_model = joblib.load('../data/congo_classifier_saved_tfidf_model.pkl') 
log_model = joblib.load('../data/congo_classifier_saved_logit_model.pkl') 

# scribe_df = pd.read_csv('../data/the_scribe_1961_1965_volumes_processed.csv')
# scribe_df.cleaned_spacy_text = scribe_df.cleaned_spacy_text.fillna('')
# scribe_filename = '../data/the_scribe_1961_1965_ao_'
bull_df = pd.read_csv('../data/Afro_Asian_Bulletin_1961_1967_volumes_processed.csv')
bull_filename = '../data/Afro_Asian_Bulletin_1961_1967_ao_'
bull_df.cleaned_spacy_text = bull_df.cleaned_spacy_text.fillna('')
classify_corpus(bull_df, tfidf_model, log_model, bull_filename)

arab_review_df = pd.read_csv('../data/Arab_Review_1960_1963_volumes_processed.csv')
arab_review_filename = '../data/Arab_Review_1960_1963_ao_'
arab_review_df.cleaned_spacy_text = arab_review_df.cleaned_spacy_text.fillna('')
classify_corpus(arab_review_df, tfidf_model, log_model, arab_review_filename)


ecopol_df = pd.read_csv('../data/Egyptian_Economic_and_Political_Review_1954_1962_volumes_processed.csv')
ecopol_filename = '../data/Egyptian_Economic_and_Political_Review_1954_1962_ao_'
ecopol_df.cleaned_spacy_text = ecopol_df.cleaned_spacy_text.fillna('')
classify_corpus(ecopol_df, tfidf_model, log_model, ecopol_filename)

liberator_df = pd.read_csv('../data/Liberator_1961_1971_volumes_processed.csv')
liberator_filename = '../data/Liberator_1961_1971_ao_'
liberator_df.cleaned_spacy_text = liberator_df.cleaned_spacy_text.fillna('')
classify_corpus(liberator_df, tfidf_model, log_model, liberator_filename)

freedomways_df = pd.read_csv('../data/Freedomways_Scraped_1961_1985_all_issues_texts_cleaned.csv')
freedomways_filename = '../data/Freedomways_Scraped_1961_1985_ao_'
freedomways_df.cleaned_spacy_text = freedomways_df.cleaned_spacy_text.fillna('')
classify_corpus(freedomways_df, tfidf_model, log_model, freedomways_filename)

