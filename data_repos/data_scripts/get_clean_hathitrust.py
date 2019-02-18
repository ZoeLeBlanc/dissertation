import pandas as pd 
from htrc_features import FeatureReader
import os, sys
import glob
import numpy as np 
from progress.bar import IncrementalBar

def read_collections(metadata, folder):
    '''This function reads in the metadata of a collection created on Hathi Trust and the folder destination. It gets the volume and tokenlist from Hathi Trust, and then calls spread table which separates out by page all tokens.'''

    md = pd.read_csv(metadata,sep='\t')
    volids = md['htitem_id'].tolist()
    print(volids)
    fr = FeatureReader(ids=volids)
    for vol in fr:
        row = md.loc[md['htitem_id'] == vol.id].copy()
        title = row['title'].values[0]
        title = title.lower().replace('.', '').replace('&amp;', 'and').replace('/', ' ').replace('-', ' ').split(" ")
        title = "_".join(title)
        title= folder+title
        file_name = title + '.csv'
        print(file_name, folder)
        a = vol.tokenlist(pos=False, case=False, section='all')
        a.to_csv(file_name)
        spread_table(title, file_name)

def read_ids(folder):
    '''This function reads in the metadata of a collection created on Hathi Trust and the folder destination. It gets the volume and tokenlist from Hathi Trust, and then calls spread table which separates out by page all tokens.'''
    
    
    volids = []
    print(volids)
    fr = FeatureReader(ids=volids)
    for vol in fr:
        title = vol.title
        title = title.lower().replace('.', '').replace('&amp;', 'and').replace('/', ' ').replace('-', ' ').split(" ")
        title = "_".join(title)
        title= folder+title
        file_name = title + '.csv'
        print(file_name, folder)
        # a = vol.tokenlist(pos=False, case=False, section='all')
        # a.to_csv(file_name)
        # spread_table(title, file_name)

def spread_table(title, file_name):
    
    df = pd.read_csv(file_name)
    pages = df['page'].unique()
    final_df = df[0:0]
    get_data = IncrementalBar('spreading table', max=len(pages))
    for i, page in enumerate(pages):
        get_data.next()
        page_df = df[0:0]
        selected_rows = df.loc[df['page'] == page].copy()
        for index, row in selected_rows.iterrows():
            token_count = row['count']
            for i in range(0, token_count):
                page_df = page_df.append(row, ignore_index=True)
        final_df = final_df.append(page_df, ignore_index=True)
    get_data.finish()
    final_df = final_df.drop(columns=['section', 'count'])

    final_df['lowercase'] = final_df['lowercase'].astype(str)
    groupby_df = final_df.groupby('page')['lowercase'].apply(' '.join).reset_index()
    final_df = final_df.drop_duplicates(subset=['page'], keep='first')
    final_df = final_df.drop(columns='lowercase')
    final = pd.merge(final_df, groupby_df, on='page', how='outer')
    final_df = final[['page', 'lowercase']]
    # print(final_df)
    final_df.to_csv(title + '_grouped.csv')

if __name__ ==  "__main__" :
	read_collections('../data_sources/hathi_trust_metadatas/EgyptianEconomicandPoliticalReviewMetaData.txt', '../data_sources/Egyptian_Economic_and_Political_Review_HathiTrust/')