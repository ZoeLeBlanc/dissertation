from bs4 import BeautifulSoup
import pandas as pd
import requests
import os
from xml.sax import saxutils as su
import nltk
import spacy
# nltk.download('punkt')
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
# nltk.download('stopwords')
from progress.bar import IncrementalBar
import warnings
warnings.filterwarnings('ignore')
nlp = spacy.load('en_core_web_lg')
'''scraped from http://voices.revealdigital.com/cgi-bin/independentvoices?a=cl&cl=CL1&sp=IBJBJF&ai=1&e=-------en-20--1--txt-txIN---------------1'''



def custom_tokenize(text):
    if not text:
#       print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    return nltk.word_tokenize(text)

def process_text(df):
    df_1 = df[0:0]
    processing = IncrementalBar('processing text', max=len(df.index))
    for index, row in df.iterrows():
        processing.next()
        raw_text = row['texts']

        tokens = custom_tokenize(raw_text)
        row['original_text'] = raw_text
        row['cleaned_nltk_text'] = ''
        row['cleaned_spacy_text'] = ''
        row['original_counts'] = len(row.texts.split())
        row['cleaned_nltk_counts'] = 0
        row['cleaned_spacy_counts'] = 0
        page_terms = ''
        for t in tokens:
            if (t.lower() not in string.punctuation) & (t.lower() not in stopwords.words('english')):
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



def get_issue(df):

    '''This function scrapes volume links from a Hathi Trust record page in case the collection making is not working.'''
    df = pd.read_csv(df)
    headers={"X-Requested-With":"XMLHttpRequest","User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36"}
    final_df = []
    output_path = '../data_sources/Freedomways_Scraped_1961_1985/Freedomways_Scraped_1961_1985_all_issues_texts.csv'
    for index, row in df.iterrows():
        link = row.issue_link
        result = requests.get(link, headers=headers)
        ht_page = result.content
        soup = BeautifulSoup(ht_page, 'lxml')
        soup = BeautifulSoup(ht_page, 'html.parser')
        text = soup.text
        soup2 = BeautifulSoup(text, 'html.parser')
        links = soup2.find_all('a')
        pages = []
        for link in links:
            link = link.attrs['href']
            split_l = link.split('&')
            la = 'http://voices.revealdigital.com/'+split_l[0]+'a&command=getSectionText&'+split_l[1]+'&f=XML&'+split_l[2]
 
            result1 = requests.get(la, headers=headers)
            ht_page2 = result1.content

            soup3 = BeautifulSoup(ht_page2, 'lxml')
            text1 = soup3.text
            soup4 = BeautifulSoup(text1, 'html.parser')
            texts = ''
            ps = soup4.findAll('p')
            
            for p in ps:
                texts += p.getText() +' '
            page = {
                'texts':texts,
                'page_link': link,
                'text_link':la,
                'issue': row.issue_text
            }
            pages.append(page)
        issue_df = pd.DataFrame.from_dict(pages, orient='columns')
        print(len(issue_df))
        if os.path.exists(output_path):
            issue_df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            issue_df.to_csv(output_path, header=True, index=False)


def get_all_issues(page, file_name):
    '''This function scrapes volume links from a Hathi Trust record page in case the collection making is not working.'''

    headers={"X-Requested-With":"XMLHttpRequest","User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36"}
    result = requests.get(page, headers=headers)
    ht_page = result.content
    # soup = BeautifulSoup(ht_page, 'lxml')
    soup = BeautifulSoup(ht_page, 'html.parser')
    # text = soup.text
    # soup2 = BeautifulSoup(text, 'html.parser')
    links = soup.find('ul')
    links = links.findAll('a')
    df = []
    for l in links:
            split_l = l.attrs['href'].split('&')
            # print(len(split_l), l.attrs['href'])
            la = 'http://voices.revealdigital.com/'+split_l[0]+'a&command=getDocumentContents&'+split_l[1]+'&f=XML&'+split_l[2]
            issue = {
            'issue_text':l.getText().replace('\n',' '),
            'issue_link':la,
            }
            df.append(issue)
    final_df = pd.DataFrame.from_dict(df, orient='columns')
    final_df.to_csv(file_name)
    print(final_df)
    get_issue(file_name)

def clean_dataframe(file_name):
    
    df = pd.read_csv(file_name)
    df = df.dropna(subset=['texts'])
    df = df.drop_duplicates()
    df.reset_index(drop=True, inplace=True)
    df['page_number'] = df.index
    dates = ['1961' , '1962','1963']
    seasons = {
        'Spring': 'April',
        '(First':'April',
        'Summer': 'July',
        '(Second':'July',
        'Fall':'October',
        '(Third':'October',
        'Winder':'January',
        '(Fourth': 'January'
    }
    df['date'] = ''
    df['year'] = ''
    df['month'] = ''
    df['day'] = '01'
    df['vol'] = ''
    df['issue_no'] = ''
    df.date = df.issue.str.split(',')
    df.year = df.date.str[0].str.split(' ').str[-1]
    df.month = df.date.str[0].str.split(' ').str[1]
    df.month = df.month.map(seasons)
    df.vol = df.date.str[1]
    df.issue_no = df.date.str[2]
    df = df.drop('date', 1)
    df = process_text(df)

    df.to_csv('../data_sources/Freedomways_Scraped_1961_1985/Freedomways_Scraped_1961_1985_all_issues_texts_cleaned.csv')



clean_dataframe('../data_sources/Freedomways_Scraped_1961_1985/Freedomways_Scraped_1961_1985_all_issues_texts.csv')
# get_all_issues('http://voices.revealdigital.com/cgi-bin/independentvoices?a=cl&cl=CL1&sp=IBJBJF&ai=1&e=-------en-20--1--txt-txIN---------------1', '../data_sources/Freedomways_Scraped_1961_1985/Freedomways_Scraped_1961_1985_all_issues_links.csv')
# get_issue('../data_sources/Freedomways_Scraped_1961_1985/Freedomways_Scraped_1961_1985_all_issues_links.csv')

