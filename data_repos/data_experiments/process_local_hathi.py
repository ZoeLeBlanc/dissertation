import spacy
import pandas as pd
import os, sys
import glob
import nltk
# nltk.download('punkt')
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
# nltk.download('stopwords')
from progress.bar import IncrementalBar
import warnings
warnings.filterwarnings('ignore')
nlp = spacy.load('en_core_web_lg')

def custom_tokenize(text):
    if not text:
#       print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    return nltk.word_tokenize(text)

def process_text(df, stopping):
    df_1 = df[0:0]
    processing = IncrementalBar('processing text', max=len(df.index))
    for index, row in df.iterrows():
        processing.next()
        raw_text = row['lowercase']
        row['tokenized_text'] = ''
        row['spacy_text'] = ''
        row['tokenized_counts'] = 0
        row['spacy_counts'] = 0
        tokens = custom_tokenize(raw_text)
        page_terms = ''
        for t in tokens:
            if stopping:
                if t in string.punctuation:
                    continue
                elif t in stopwords.words('english'):
                    continue
                else:
                    page_terms += t.lower() + ' '
            else:
                    page_terms += t.lower() + ' '
        row.tokenized_text = page_terms
        sent_terms = ''
        spacy_text = nlp(page_terms)
        for ent in spacy_text.ents:
            if '.' in ent.text:
                text = ('').join(ent.text.split('.'))
                sent_terms += text + ' '
            else:
                sent_terms += ent.text + ' '
        row.spacy_text = sent_terms

        df_1 = df_1.append(row, ignore_index=True)    
    df_1.reset_index(inplace=True)
    processing.finish()
    df_1.tokenized_counts = df_1.tokenized_text.str.split().str.len()
    df_1.spacy_counts = df_1.spacy_text.str.split().str.len()
    return df_1


def get_hathi(stopping, aggregating):
    df_hathi = pd.read_csv('../data/hathi_trust_1963_1966_stopwords.csv')

    
    data = process_text(df_hathi, stopping)
    data = data.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'tokenized', 'word_count', 'token_counts', 'years', 'months'], axis=1)
    data.to_csv('../data/hathi_trust_1963_1966_stopwords_cleaned.csv')

    if aggregating: 
        df_grouped_spacy = data.groupby(['htrc_vol', 'date', 'year', 'first_month', 'second_month', 'first_month_index', 'second_month_index'])['spacy_text'].apply(' '.join).reset_index()
        
        df_grouped_spacy_counts = data.groupby(['htrc_vol', 'date', 'year', 'first_month', 'second_month', 'first_month_index', 'second_month_index'])['spacy_counts'].sum().reset_index()
        df_grouped_tokenized = data.groupby(['htrc_vol', 'date', 'year', 'first_month', 'second_month', 'first_month_index', 'second_month_index'])['tokenized_text'].apply(' '.join).reset_index()
        df_grouped_tokenized_counts = data.groupby(['htrc_vol', 'date', 'year', 'first_month', 'second_month', 'first_month_index', 'second_month_index'])['tokenized_counts'].sum().reset_index()
        df_grouped_pages = data.groupby(['htrc_vol', 'date', 'year', 'first_month', 'second_month', 'first_month_index', 'second_month_index'])['page'].count().reset_index()
        final_df = pd.merge(df_grouped_spacy, df_grouped_tokenized, on=['htrc_vol', 'date', 'year', 'first_month', 'second_month', 'first_month_index', 'second_month_index'])
        df_1 = pd.merge(final_df, df_grouped_spacy_counts, on=['htrc_vol', 'date', 'year', 'first_month', 'second_month', 'first_month_index', 'second_month_index'])
        df_2 = pd.merge(df_1, df_grouped_tokenized_counts, on=['htrc_vol', 'date', 'year', 'first_month', 'second_month', 'first_month_index', 'second_month_index']) 
        data = pd.merge(df_2, df_grouped_pages, on=['htrc_vol', 'date', 'year', 'first_month', 'second_month', 'first_month_index', 'second_month_index']) 
        data.to_csv('../data/hathi_trust_1963_1966_stopwords_aggregating.csv')
    

if __name__ ==  "__main__" :
    print(len(sys.argv))
    stopping = False
    aggregating = False
    if len(sys.argv) > 1:
        stopping = sys.argv[1]
        aggregating = sys.argv[2]
    get_hathi(True, True)
