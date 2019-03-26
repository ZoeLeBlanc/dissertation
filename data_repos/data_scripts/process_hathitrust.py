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

def process_text(df, title, dates, vols):
    df_1 = df[0:0]
    processing = IncrementalBar('processing text', max=len(df.index))
    for index, row in df.iterrows():
        processing.next()
        raw_text = row['lowercase']
        row['title'] = title
        row['dates'] = dates
        row['vols'] = vols

        tokens = custom_tokenize(raw_text)
        row['original_text'] = raw_text
        row['cleaned_nltk_text'] = ''
        row['cleaned_spacy_text'] = ''
        row['original_counts'] = len(row.lowercase.split())
        row['cleaned_nltk_counts'] = 0
        row['cleaned_spacy_counts'] = 0
        page_terms = ''
        for t in tokens:
            if t.lower() in string.punctuation:
                continue
            elif t.lower() in stopwords.words('english'):
                continue
            else:
                page_terms += t.lower() + ' '
        row.cleaned_nltk_text = page_terms
        spacy_terms = ''
        spacy_text = nlp(page_terms)
        for ent in spacy_text:
            if len(ent.ent_type_) > 0 or ent.is_alpha:
                if( ent.is_punct == False) and (any(i.isdigit() for i in ent.text) == False) and (ent.is_stop ==False):
                    if '.' in ent.text:
                        text = ('').join(ent.text.split('.'))
                        spacy_terms += text + ' '
                    else:
                        spacy_terms += ent.text + ' '
        row.cleaned_spacy_text = spacy_terms
        df_1 = df_1.append(row, ignore_index=True)    
    df_1.reset_index(inplace=True)
    processing.finish()
    df_1.cleaned_nltk_counts = df_1.cleaned_nltk_text.str.split().str.len()
    df_1.cleaned_spacy_counts = df_1.cleaned_spacy_text.str.split().str.len()
    return df_1


def get_hathi(directory, output_path):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if 'grouped' in file:
                # print(file, directory)
                df_hathi = pd.read_csv(directory + file)
                hathi_vol = file.split('/')[-1].split('_grouped')[0]
                print(hathi_vol.split('_'))
                title = ('_').join(hathi_vol.split('_')[:-2])
                dates = hathi_vol.split('_')[-1]
                vols = hathi_vol.split('_')[-2]
                # if len(hathi_vol.split('_')) >3:
                #     vols = ('_').join(hathi_vol.split('_')[1:3])
                # # vols = 'na'
                print(hathi_vol, title, dates, vols)
                data = process_text(df_hathi, title, dates, vols)
                data = data.drop(['Unnamed: 0'], axis=1)
                if os.path.exists(output_path):
                    data.to_csv(output_path, mode='a', header=False, index=False)
                else:
                    data.to_csv(output_path, header=True, index=False)
 

if __name__ ==  "__main__" :
    get_hathi('../data_sources/arab_affairs_1960_1962_HathiTrust/', '../data_sources/arab_affairs_1960_1962_HathiTrust/arab_affairs_1960_1962_volumes_processed.csv')
    
