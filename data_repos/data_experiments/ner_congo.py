import altair as alt
import pandas as pd
import glob, os
import spacy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.layouts import row, column
from bokeh.io import  output_notebook
import nltk
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
from progress.bar import IncrementalBar

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../data/final_ner_congo_identified.csv')
output_path = '../data/final_cleaned_matrix_data_ner_congo.csv'
print(df.columns)
from itertools import combinations 
def get_terms(rows):
    terms = []
    for index, r in rows.iterrows():
        # terms.append(r.term)
        for x in range(r.word_counts):
            terms.append(r.term)
            
    return terms
fixed_df = df.groupby(['date','year', 'month', 'day', 'page_number', 'binned']).apply(get_terms).reset_index()
fixed_df.rename(columns={0:'terms'}, inplace =True)
print(fixed_df.columns)
processing = IncrementalBar('processing text', max=len(fixed_df.index))
from itertools import combinations 
for index, row in fixed_df.iterrows():
    processing.next()
    new_df = {}
    new_df['pn'] = row.page_number
    new_df['year'] = row.year
    new_df['month'] = row.month
    new_df['day'] = row.day
    new_df['date'] = row.date
    new_df['binned'] = row.binned
    t = list(combinations(row.terms, 2))
    print(len(t), len(row.terms))
    if len(t) > 0:
        for item in t:
            new_df['source'] = item[0]
            new_df['target'] = item[1]
            
            dl = pd.DataFrame().append(new_df, ignore_index=True)
            if os.path.exists(output_path):
                dl.to_csv(output_path, mode='a', header=False, index=False)
            else:
                dl.to_csv(output_path, header=True, index=False)
processing.finish()