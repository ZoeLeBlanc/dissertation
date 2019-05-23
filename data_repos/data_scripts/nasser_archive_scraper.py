from bs4 import BeautifulSoup
import pandas as pd
import requests
import os
from xml.sax import saxutils as su
from google.cloud import translate
import google.auth
# import nltk
# import spacy
# # nltk.download('punkt')
# import string
# from nltk.corpus import stopwords
# from nltk.util import ngrams
# # nltk.download('stopwords')
# from progress.bar import IncrementalBar
# import warnings
# warnings.filterwarnings('ignore')
# nlp = spacy.load('en_core_web_lg')
_, _ = google.auth.default()
import time
df = pd.read_csv('../data_sources/nasser_archive/nasser_speeches_all.csv')
# print(len(df))
# df = df[0:1084]
# # df = df[['texts', 'title', 'text_link', 'date',
# #        'page_link', 'translated_text']]
# df.to_csv('../data_sources/nasser_archive/nasser_speeches_all.csv')
# print(len(df), df[-1:])

def translate_text(text):
    try:
        final_text = []
        for t in text:
            time.sleep(20)
            translate_client = translate.Client()
            result = translate_client.translate(t, target_language='en', source_language='ar')
            final_text.append(result['translatedText'])
        finals = (' ').join(final_text)
        print(len(finals))
        return finals
    except:
        print('error', len(text))
        return ''


def split_text(text):
    texts = text.split(' ')
    print(len(text), len(texts))
    val = round( len(texts) /round(len(text) / len(texts)))
    print(val)
    final_texts = []
    for i in range(0, len(texts), val):
        # print(i, val)
        chunk = ' '.join(texts[i:i + val])
        if i + val + val > len(texts):
            chunk = ' '.join(texts[i:])
            final_texts.append(chunk)
            break
        final_texts.append(chunk)
    print(len(final_texts))
    return final_texts


for i in range(1084, 1300):
    print(i)
    time.sleep(20)

    headers={"X-Requested-With":"XMLHttpRequest","User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36"}
    # result = requests.get(page, headers=headers)
    result_header = requests.get('http://nasser.org/Speeches/Sound.aspx?SID={}&lang=en'.format(str(i)), headers=headers)

    header_page = result_header.content
    # soup = BeautifulSoup(ht_page, 'lxml')
    soup_header = BeautifulSoup(header_page, 'html.parser')
    title = soup_header.find('span', {'id':'Title'}).getText()
    date = soup_header.find('span', {'id':'Date'}).getText()
    url = 'http://nasser.org/Speeches/html.aspx?SID={}&lang=en'.format(str(i))
    result = requests.get(url, headers=headers)

    ht_page = result.content
    # soup = BeautifulSoup(ht_page, 'lxml')
    soup = BeautifulSoup(ht_page, 'html.parser')
    # print(soup)
    link = soup.find('iframe').attrs['src']
    result2 = requests.get(link, headers=headers)
    result2.encoding = 'UTF-8'
    page2 = result2.content
    soup2 = BeautifulSoup(page2, 'html.parser')
    text = soup2.getText()
    if len(text) > 10000:
        text1 = split_text(text)
        translated_text = translate_text(text1)
    else: 
        text1 = [text]
        translated_text = translate_text(text1)
    
    page = {
                'texts':text,
                'title': title,
                'text_link':link,
                'date': date,
                'page_link':url,
                'translated_text': translated_text,
            }
    # print(page)
    page_df = pd.DataFrame(page, index=[0])
    output_path = '../data_sources/nasser_archive/nasser_speeches_all.csv'
    if os.path.exists(output_path):
        page_df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        page_df.to_csv(output_path, header=True, index=False)