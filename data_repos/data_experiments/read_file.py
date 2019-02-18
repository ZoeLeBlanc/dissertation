import pandas as pd 
from htrc_features import FeatureReader
from htrc import workset
import os
import glob
import numpy as np 
# scribe_htrc = pd.read_csv('vol_page_freq.csv')
# pages = scribe_htrc['page'].unique()
# # final_df = scribe_htrc[0:0]
# fr = FeatureReader(ids=["inu.30000125592232"])
# title = ''
# for vol in fr:
#     title = vol.title.lower().replace('.', '').split(" ")
#     title = "_".join(title)
# print(title)
# for i, page in enumerate(pages):
#     df = scribe_htrc[0:0]
#     selected_rows = scribe_htrc.loc[scribe_htrc['page'] == page].copy()
#     for index, row in selected_rows.iterrows():
#         token_count = row['count']
#         for i in range(0, token_count):
#             df = df.append(row, ignore_index=True)
        
#     df.to_csv('./htrc_pages/'+str(page)+'_'+title+'.csv')
# test = pd.read_csv('term_page_freq.csv')
# print(test[0:2])
# result = [i for i in glob.glob('*.{}'.format('csv'))]
# # final_df = pd.DataFrame(columns=['page', 'lowercase', 'counts'], index=None)
# output_path = 'final_htrc.csv'
# for filename in result:
#     if os.path.exists(output_path):
#         df = pd.read_csv(filename, index_col=False)
#         df.to_csv(output_path, mode='a', header=False, index=False)
#     else:
#         df = pd.read_csv(filename, index_col=False)
#         df.to_csv(output_path, header=True, index=False)
volids = workset.load_hathitrust_collection('https://babel.hathitrust.org/cgi/mb?a=listis&c=648138425')
fr = FeatureReader(ids=volids)
# # print(fr)
# fr = FeatureReader(ids=["inu.30000125592232"])
# # # # print(fr)

# # # # # final_df = pd.DataFrame(columns=['page', 'character', 'frequency'], index=None)
# # # # output_path = 'htrc_test.csv'
for index, vol in enumerate(fr):
# #     a = vol.tokens_per_page()
# #     print(a)
    # a = vol.tokenlist(pos=False, case=False, section='all')
    # a.to_csv(vol.title + str(index) + '_vol_page_freq.csv')
    # print("Volume %s is a %s page text written in %s. You can doublecheck at %s" % (vol.id, vol.page_count, vol.language, vol.handle_url))
    # print(vol.metadata)
    print(vol.title)
# #     print(vol.metadata['published'][0])

# #     a = vol.tokens_per_page()
# #     print(a)
#     val = 0
    # for page in vol.pages():
    #     print(page, len(page.tokens()))
#         one_df = pd.DataFrame(columns=['page', 'character', 'frequency'], index=None)
#         df = page.tokenlist(pos=False, case=False, section='group')
#         print(df.columns)
#         if len(df.index) > 1:
#             df.to_csv(str(val)+'page.csv')
#             print(df.head(1))
#             val = val +1

# final_df.reset_index(inplace=True, drop=True)
# final_df.to_csv('htrc_test_fulldf.csv')
# text = pd.read_csv('htrc-ef-all-files.txt', error_bad_lines=False)
# print(text[0:1])