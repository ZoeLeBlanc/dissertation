import pandas as pd
import os, sys
import glob
import nltk
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
import warnings
warnings.filterwarnings('ignore')

'''This script aggregates the data from Image Lucida and the Hathi Trust Full Text Corpus into one giant csv'''

def custom_tokenize(text):
    '''Tokenize each page in an issue'''
    if not text:
        text = ''
    return nltk.word_tokenize(text)

def process_hathi_text(text, stopping):
    '''Process text with tokenizing and spacy nlp'''
    tokens = custom_tokenize(text)
    page_terms = ''
    terms = []
    for t in tokens:
        if stopping:
            if t in string.punctuation:
                continue
            elif t in stopwords.words('english'):
                continue
            else:
                page_terms += t.lower() + ' '
                terms.append(t.lower())
        else:
            page_terms += t.lower() + ' '
            terms.append(t.lower())
    counts = len(terms)
    return page_terms, counts

def get_hathi_files(dir, output_file, stopping, aggregating, metadata):
    '''Get all hathi text files and put them into csv for aggregating'''
    cwd = os.getcwd()
    md = pd.read_table(metadata)
    for subdir, dirs, files in os.walk(dir):
        os.chdir(dir)
        for f in files:
            if 'volume' not in f:
                page_number = int(f.split('.')[0])
                with open(subdir+ '/'+f, 'r') as file:
                    text = file.read()
                    tokenized_text, tokenized_counts = process_hathi_text(text, stopping)
                    df = {}
                    df['tokenized_text'] = tokenized_text
                    df['tokenized_counts'] = tokenized_counts
                    df['page_number'] = page_number
                    df['vol'] = subdir.split('/')[-1]
                    row = md.loc[md['htitem_id'] == df.vol].copy()
                    title = row['title'].values[0]
                    df['title'] = ('_').join(title.lower().replace('.', '').split(" "))
                    d = df
                    df = pd.DataFrame().append(d, ignore_index=True)

                    os.chdir(cwd)
                    if os.path.exists(output_file):
                        df.to_csv(output_file, mode='a', header=False, index=False)
                    else:
                        df.to_csv(output_file, header=True, index=False)
    # if aggregating:
    #     df = pd.read_csv(output_file)
    #     df_grouped_counts = df.groupby(['title', 'vol'])['tokenized_counts'].sum().reset_index()
    #     df_grouped_text = df.groupby(['title', 'vol'])['tokenized_text'].apply(' '.join).reset_index()
    #     df_grouped_pages = df.groupby(['title', 'vol'])['page_number'].count().reset_index()
    #     df_1 = pd.merge(df_grouped_counts, df_grouped_text, on=['title', 'vol'])
    #     final_df = pd.merge(df_1, df_grouped_pages, on=['title', 'vol'])
    #     final_df.to_csv(output_file.split('.')[0] 
    #     +'_grouped.csv')

def get_hathi_files_with_aggregation(dir, output_file, stopping, metadata):
    '''Get all hathi text files and put them into csv for aggregating. ASSUMES that I ran the download function with the -c flag to concat all issues within a directory'''
    cwd = os.getcwd()
    md = pd.read_table(metadata)
    for subdir, dirs, files in os.walk(dir):
        os.chdir(dir)
        for f in files:
            if 'volume' not in f:
                page_number = int(f.split('.')[0])
                with open(dir+ '/'+f, 'r') as file:
                    text = file.read()
                    tokenized_text, tokenized_counts = process_hathi_text(text, stopping)
                    df = {}
                    df['tokenized_text'] = tokenized_text
                    df['tokenized_counts'] = tokenized_counts
                    df['page_number'] = page_number
                    df['vol'] = f.split('.txt')[0]
                    row = md.loc[md['htitem_id'] == df.vol].copy()
                    title = row['title'].values[0]
                    df['title'] = title.lower().replace('.', '').split(" ")
                    d = df
                    df = pd.DataFrame().append(d, ignore_index=True)

                    os.chdir(cwd)
                    if os.path.exists(output_file):
                        df.to_csv(output_file, mode='a', header=False, index=False)
                    else:
                        df.to_csv(output_file, header=True, index=False)
        
if __name__ ==  "__main__" :
    metadata = 'fewer_ao_metadata.txt'
    stopping = True
    aggregating = True
    direct = '/media/secure_volume/workset/'
    output = 'arab_observer_1963_1966_hathi.csv'
    get_hathi_files(direct, output, stopping, aggregating, metadata)
