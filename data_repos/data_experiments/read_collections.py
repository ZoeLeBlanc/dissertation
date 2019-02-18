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
    print(ids)
    volids = workset.load_hathitrust_collection('https://babel.hathitrust.org/cgi/mb?a=listis;c=1885558567')
    print(volids)
    https://babel.hathitrust.org/cgi/mb?a=listis;c=648138425
    # print('inu.32000013025095', 'inu.32000013025087', 'inu.32000013025079', 'inu.32000013025061', 'inu.32000013025053', 'inu.32000013025046', 'mdp.39015056038089', 'inu.32000013024635', 'inu.32000013024627', 'mdp.39015056038071', 'mdp.39015056038246', 'mdp.39015056038253', 'inu.32000013025194', 'inu.32000013025160', 'mdp.39015056038063', 'mdp.39015056038238', 'mdp.39015056038402', 'mdp.39015056038410', 'inu.32000013025152', 'inu.32000013025145', 'inu.32000013025137', 'inu.32000013025129', 'mdp.39015056038220', 'mdp.39015056038386', 'mdp.39015056038394', 'inu.32000013025111', 'inu.32000013025103', 'mdp.39015056038212', 'mdp.39015056038378', 'mdp.39015056038204', 'mdp.39015056038352', )
    # volids= ['uva.x030696874', 'uva.x030696873', 'uva.x030696872', 'uva.x030696871', 'uva.x030696870', 'uva.x030696869', 'uva.x030696867', 'uva.x030696866', 'uva.x030696865', 'uva.x030696864', 'uva.x030696833', 'uva.x030696834', 'uva.x030696835', 'uva.x030696836', 'uva.x030696837', 'uva.x030696838', 'uva.x030696839', 'uva.x030696840', 'uva.x030696841', 'uva.x030696843', 'uva.x030696844', 'uva.x030696845', 'uva.x030696848', 'uva.x030696849', 'uva.x030696850', 'uva.x030696851', 'uva.x030696852', 'uva.x030696854', 'uva.x030696855', 'uva.x030696856', 'uva.x030696857', 'uva.x030696859', 'uva.x030696858', 'uva.x030696860', 'uva.x030696861', 'uva.x030696862', 'uva.x030696863', 'uva.x030696876', 'uva.x030696877', 'uva.x030696878', 'uva.x030696879', 'uva.x030696880', 'uva.x030696881', 'uva.x030696882', 'uva.x030696883', 'uva.x030696884', 'uva.x030696885', 'uva.x030696886', 'uva.x030696887', 'uva.x030696888', 'uva.x030696889', 'uva.x030696890', 'uva.x030696891', 'uva.x030696892', 'uva.x030696895', 'uva.x030696896', 'uva.x030696897', 'uva.x030696898', 'uva.x030696899', 'uva.x030696900', 'uva.x030696901', 'uva.x030696902', 'uva.x030696903', 'uva.x030696904', 'inu.30000081508032', 'inu.30000122990637', 'uva.x030696905', 'uva.x030696906', 'uva.x030696907', 'uva.x030696908', 'uva.x030696909', 'uva.x030696910', 'uva.x030696911', 'uva.x030696912', 'uva.x030696913', 'uva.x030696914', 'uva.x030696915', 'uva.x030696916', 'uva.x030696917', 'uva.x030696918', 'uva.x030696919', 'uva.x030696920', 'uva.x030696921', 'uva.x030696923', 'uva.x030696924', 'uva.x030696926', 'uva.x030696927', 'uva.x030696928', 'uva.x030696929', 'uva.x030696930', 'uva.x030696931', 'uva.x030696932', 'uva.x030696934', 'uva.x030696935', 'uva.x030696936', 'uva.x030696937', 'uva.x030696938', 'uva.x030696940', 'uva.x030696941', 'uva.x030696942', 'uva.x030696943', 'uva.x030696944', 'uva.x030696945', 'uva.x030696946', 'uva.x030696947', 'uva.x030696948', 'uva.x030696949', 'uva.x030697028', 'uva.x030697029', 'uva.x030697030', 'uva.x030697031', 'uva.x030697032', 'uva.x030697033', 'uva.x030697034', 'uva.x030697035', 'inu.30000081508123', 'inu.30000081508115', 'inu.30000122990629']
    volids = ['uva.x030696874', 'uva.x030696873', 'uva.x030696872']
    fr = FeatureReader(ids=volids)
    for vol in fr:
        row = md.loc[md['htitem_id'] == vol.id].copy()
        title = row['title'].values[0]
        title = title.lower().replace('.', '').split(" ")
        title = "_".join(title)
        file_name = title + '.csv'
        a = vol.tokenlist(pos=False, case=False, section='all')
        a.to_csv(file_name)
        spread_table(title, file_name)

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
	print(sys.argv[1])
	read_collections(sys.argv[1])