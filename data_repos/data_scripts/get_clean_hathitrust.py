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

def read_ids(metadata, folder, df):
    '''This function reads in the list of ids scraped from Hathi Trust and the folder destination. It gets the volume and tokenlist from Hathi Trust, and then calls spread table which separates out by page all tokens.'''
    directory = os.path.dirname(folder)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if df:
        md = pd.read_csv(metadata)
        volids = md['vol_id'].tolist()
        # volids = [ vol for vol in volids if vol != 'uva.x030697132']
        # idx = volids.index('uva.x030697132') # for '../data_sources/hathi_trust_metadatas/middle_east_news_economic_weekly_1964_1973_008564927.csv' remove volume that is throwing error
        # volids = volids[idx+1:]
        
        fr = FeatureReader(ids=volids)
        for vol in fr:
            row = md.loc[md['vol_id'] == vol.id].copy()
            title = vol.title
            title = title.lower().split(" ")
            title = "_".join(title)+'_'+str(row.volume.values[0])+'_'+str(row.date.values[0])
            print(title)
            
            title= folder+title
            file_name = title + '.csv'
            a = vol.tokenlist(pos=False, case=False, section='all')

            file_name = file_name
            a.to_csv(file_name)
            spread_table(title, file_name)
    else:
        text_file = open(metadata, "r")
        volids = text_file.read().split('\n')
        volids = [vol for vol in volids if len(vol) > 0]
        volids = volids[:-1]
        volids = volids[0:3]
        
        fr = FeatureReader(ids=volids)
        for vol in fr:
            title = vol.title + ' ' + vol.pub_date
            print(vol.metadata)
            title = title.lower().replace('.', '').replace('&amp;', 'and').replace('/', ' ').replace('-', ' ').split(" ")
            title = "_".join(title)
            print(title)
            title= folder+title
            file_name = title + '.csv'
            print(file_name, folder)
            a = vol.tokenlist(pos=False, case=False, section='all')
            a.to_csv(file_name)
            spread_table(title, file_name)


def spread_table(title, file_name):
    print('spread', type(title), title, type(file_name), file_name)
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
    final_df.to_csv(title + '_grouped.csv')

if __name__ ==  "__main__" :
	# read_collections('../data_sources/hathi_trust_metadatas/EgyptianEconomicandPoliticalReviewMetaData.txt', '../data_sources/Egyptian_Economic_and_Political_Review_HathiTrust/')
    read_ids('../data_sources/hathi_trust_metadatas/nashrat_akhbar_jamiat_al-Duwal_al-Arabiyah_1962_1967_003839852.csv', '../data_sources/nashrat_akhbar_jamiat_al-Duwal_al-Arabiyah_1962_1967_HathiTrust/', True)