import altair as alt
import pandas as pd
import glob, os, sys
import spacy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import MDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_selection import chi2
import gensim
from gensim.corpora import Dictionary
from gensim.similarities import MatrixSimilarity
from gensim.models import ldamodel, doc2vec, LsiModel
import nltk
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
nlp = spacy.load('en_core_web_lg')

def find_frequencies(output_path, term):
    nltk_df = pd.read_csv('Arab_Observer_Tokenized_Distributions.csv')
    cairo_df = nltk_df
    print(cairo_df.htrc_vol.unique())
    cairo_df = cairo_df.loc[(cairo_df['lowercase'].str.contains(term) == True)]
    for index,row in cairo_df.iterrows():
        a = row['lowercase']
        words = nltk.tokenize.word_tokenize(a)
        page_terms = []
        for t in words:
            if t in string.punctuation:
                continue
            elif t in stopwords.words('english'):
                continue
            else:
                page_terms.append(t.lower())
        word_dist = nltk.FreqDist(page_terms)
        top = len(page_terms)
        result = pd.DataFrame(word_dist.most_common(top),
                        columns=['word', 'frequency'])
        result['htrc_vol'] = row.htrc_vol
        result['page'] = row.page
        df = result
        if os.path.exists(output_path):
            df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            df.to_csv(output_path, header=True, index=False)
        return visualize_freq(output_path)

def visualize_freq(output_path):
    data = pd.read_csv(output_path)
    data['year'] = data.htrc_vol.str.split('_').str[2]
    data['months'] = data.htrc_vol.str.split('_').str[3]
    data['first_month'] = data.months.str.split('-').str[0]
    data['second_month'] = data.months.str.split('-').str[1]
    data['second_month'].fillna('dec', inplace=True)
    data.first_month = data.first_month.str.capitalize()
    data.second_month = data.second_month.str.capitalize()
    data['first_month_index'] = pd.to_datetime(data['first_month'], format='%b', errors='coerce').dt.month
    data['second_month_index'] = pd.to_datetime(data['second_month'], format='%b', errors='coerce').dt.month
    data = data.sort_values(by=['year', 'first_month_index'])
    data.htrc_vol.str.split('_')
    data['date'] = pd.to_datetime(data['year'].apply(str)+'-'+data['first_month_index'].apply(str), format='%Y-%m')
    data.date =data.date.dt.strftime('%Y-%m')
    data = data[data['frequency'] > 3]
    chart = alt.Chart(data).mark_circle(
        opacity=0.8,
        stroke='black',
        strokeWidth=1
    ).encode(
        alt.X('date:O'),
        alt.Y('word', axis=alt.Axis(labelAngle=0)),
        alt.Size('frequency',
            scale=alt.Scale(range=[0, 2000]),
            legend=alt.Legend(title='counts')
        ),
        alt.Color('word', scale=alt.Scale(scheme='category20'), legend=None)
    ).properties(
        width=1400,
        height=10000
    )
    chart.serve()

if __name__ ==  "__main__" :
	# print(sys.argv[1])
    # find_frequencies(sys.argv[1], sys.argv[2])
    visualize_freq('bandung_freq.csv')