import pandas as pd 
from htrc_features import FeatureReader
from htrc import workset
import os, sys
import glob
import numpy as np 
from progress.bar import IncrementalBar

def read_collections(metadata):
    
    md = pd.read_table(metadata)
    ids = md['htitem_id'].tolist()
    # print(ids)

    fr = FeatureReader(ids=ids)
    for vol in fr:
        row = md.loc[md['htitem_id'] == vol.id].copy()
        title = row['title'].values[0]
        title = title.lower().replace('.', '').split(" ")
        title = "_".join(title)
        print(title)
        file_name = title + '.csv'
        a = vol.tokenlist(pos=False, case=False, section='all')
        a.to_csv(file_name)
        spread_table(title, file_name)

def spread_table(title, file_name):
    
    df = pd.read_csv(file_name)
    print(df.columns)
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
    print(len(df.index), len(final_df.index))
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
	print(sys.argv[1])
	read_collections(sys.argv[1])