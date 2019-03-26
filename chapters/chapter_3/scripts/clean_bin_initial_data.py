
import pandas as pd
import numpy as np

import os, sys
import glob
import nltk
# nltk.download('punkt')
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
import csv
import math
import statistics
import datetime
import itertools
from progress.bar import IncrementalBar
import warnings
warnings.filterwarnings('ignore')
### Load in data for issues and each page of all issues
df = pd.read_csv('../data/arab_observer_1960_1962_wo_stopwords.csv')
df_hathi = pd.read_csv('../data/hathi_trust_1963_1966_cleaned_wo_stopwords.csv')


df = df.drop(['Unnamed: 0', 'index', 'google_vision_text', 'file_name', 'vol'], axis=1)
df['type'] = 'collected_issues'
df_hathi = df_hathi.drop(['Unnamed: 0', 'level_0', 'htrc_vol', 'index', 'lowercase','second_month', 'second_month_index', 'first_month_index'], axis=1)

df_hathi.rename(columns={'first_month': 'month', 'page': 'page_number'}, inplace=True)
df_hathi['type'] = 'hathi_trust'

df_hathi['day'] = '01'
df_hathi['date'] = df_hathi.year.astype(str)+ '-' +df_hathi.month.astype(str) +'-' + df_hathi.day.astype(str)
df_hathi['string_date'] = df_hathi.date.astype(str)
df['string_date'] = df.date.astype(str)


df_1 = df.append(df_hathi, ignore_index=True)

df_1.month[df_1.month == 'Jan'] = 'January'
df_1.month[df_1.month == 'Aug'] = 'August'
df_1.month[df_1.month == 'Oct'] = 'October'
df_1.month[df_1.month == 'Sept'] = 'September'
df_1.month[df_1.month == 'Sep'] = 'September'
df_1.month[df_1.month == 'Apr'] = 'April'
df_1.month[df_1.month == 'Jul'] = 'July'
df_1['date'] = df_1.year.astype(str) +'-'+df_1.month+'-'+df_1.day.astype(str)
df_1['datetime'] = pd.to_datetime(df_1['date'], format='%Y%m%d', errors='ignore')
print(df_1.date.unique())
# df_1.to_csv('../data/hathi_ef/combined_il_ht_data.csv')

df_grouped = df_1.groupby(['date', 'year'])['page_number'].count().reset_index()
df_grouped_1960 = df_grouped.loc[df_grouped.year < 1963]
df_grouped_1966 = df_grouped.loc[df_grouped.year > 1962]
### Calculate bin size by using the commands below
# df_grouped_1960.page_number.plot.kde()
# df_grouped_1960.page_number.mean(), df_grouped_1960.page_number.mode(), df_grouped_1960.page_number.median()
def get_bins(rows):
    bins = rows.values[0]/36
    return bins

df_grouped_bins = df_grouped_1966.groupby(['date'])['page_number'].apply(get_bins).reset_index()
df_grouped_bins.rename(columns={'page_number': 'bins'}, inplace=True)
df_1960 = df_1.loc[df_1.year < 1963]
df_1966 = df_1.loc[df_1.year > 1962]
df_1960['bins'] = 0
df_1960['binned'] = 0
df_66 = pd.merge(df_grouped_bins, df_1966, how='left', on='date')
dates = df_66.date.unique().tolist()
frames = []
processing_dates = IncrementalBar('processing dates', max=len(dates))
for d in dates:
    processing_dates.next()
    rows = df_66.loc[df_66.date == d]
    bi = rows.bins.values[0]
    labels = list(range(0, int(bi)))
    t = pd.qcut(rows.page_number, int(bi), labels=labels)
    rows['binned'] = t
    frames.append(rows)
processing_dates.finish()
    
df1966 = pd.concat(frames)
final_df = pd.concat([df_1960, df1966])
print(final_df.columns)
final_df[['cleaned_spacy_text']] = final_df[['cleaned_spacy_text']].fillna(value='')
# final_df['token_counts'] = final_df.cleaned_spacy_texts.str.split().str.len()
# final_df['datetime'] = pd.to_datetime(final_df['date'], format='%Y-%B-%d', errors='coerce')
full_corpus_df = final_df
full_corpus_df.cleaned_spacy_text.dropna(inplace=True)
processing_text = IncrementalBar('processing text', max=len(full_corpus_df))
def custom_tokenize(text):
    if not text:
#       print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    return nltk.word_tokenize(text)
def clean_texts(row):
    # raw_text.cleaned_spacy_text.values[0]
    processing_text.next()
    tokens = custom_tokenize(row)
    page_terms = ''
    for t in tokens:
        if t in string.punctuation:
            continue
        elif t in stopwords.words('english'):
            continue
        else:
            page_terms += t.lower() + ' '
    return page_terms

full_corpus_df['really_clean'] = full_corpus_df.cleaned_spacy_text.apply(clean_texts)
processing_text.finish()
full_corpus_df = full_corpus_df.drop(['cleaned_spacy_text'], axis=1)
full_corpus_df.rename(columns={'really_clean':'cleaned_spacy_text'}, inplace=True)

final_df.to_csv('../data/arab_observer_corpus_cleaned.csv')
