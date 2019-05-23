
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


def get_most_informative_features(vectorizer, clf, file_name, n=50):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    

    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n+1):-1])
    coef_0 = pd.DataFrame(coefs_with_fns, columns = ['coef_0', 'feature_0'])
    coef_1 = pd.DataFrame(coefs_with_fns[::-1], columns = ['coef_1', 'feature_1'])
    coefs = pd.concat([coef_0, coef_1], axis=1, sort=False)
    coefs.to_csv(file_name +'_features.csv')

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


def train_model(df, file_name, full_corpus_df):
    df = shuffle(df)
    y = df['class_numb']
    category_id_df = df[['classify', 'class_numb']].drop_duplicates().sort_values('class_numb')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['classify', 'class_numb']].values)
    labels = y
    tfidf_model = TfidfVectorizer(ngram_range=(1,1), lowercase=False, max_df=0.3)
    features = tfidf_model.fit_transform(df.cleaned_spacy_text.tolist())

    features_nd = features.toarray()

    training_features, test_features, training_target, test_target = train_test_split(features_nd[0:len(df['cleaned_spacy_text'])], y,test_size=0.3)
    x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,test_size = 0.3,random_state=12)

    sm = SMOTE(sampling_strategy='auto')
    x_train_res, y_train_res = sm.fit_sample(x_train, y_train)


    log_model = LogisticRegression()
    log_model = log_model.fit(X=x_train_res, y=y_train_res)
    y_pred = log_model.predict(x_val)
    print('Validation Results')
    print(log_model.score(x_val, y_val))
    print(metrics.recall_score(y_val, y_pred, average=None ))
    print("Precision:",metrics.precision_score(y_val, y_pred, average=None ))
    print('\nTest Results')
    print(log_model.score(test_features, test_target))
    print(metrics.recall_score(test_target, log_model.predict(test_features), average=None ))
    print("Precision:",metrics.precision_score(test_target, log_model.predict(test_features), average=None))
    kfold = KFold(n_splits=10, random_state=7)
    scoring = 'accuracy'
    results = cross_val_score(log_model, x_train_res, y_train_res,scoring='accuracy', cv=kfold)
    print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

    print(metrics.classification_report(y_val, y_pred, target_names=df['classify'].unique()))
    
    conf_mat = confusion_matrix(y_val, y_pred)
    print(conf_mat)
    # fig, ax = plt.subplots(figsize=(10,10))
    # sns.heatmap(conf_mat, annot=True, fmt='d',
    #             xticklabels=category_id_df.classify.values, yticklabels=category_id_df.classify.values)
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    # plt.show()
    # fig.savefig(file_name +'_confusionmatrix.png')

    

    # Output a pickle file for the model
    

    # get_most_informative_features(tfidf_model, log_model, file_name)
    classify_corpus(full_corpus_df, tfidf_model, log_model, file_name)
    joblib.dump(log_model, file_name+'_saved_logit_model.pkl')
    joblib.dump(tfidf_model, file_name+'_saved_tfidf_model.pkl')




train_model(congo_df, congo_filename, full_corpus_df)
# train_model(time_df, time_filename, full_corpus_df)
