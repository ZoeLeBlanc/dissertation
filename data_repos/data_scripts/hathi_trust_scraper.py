from bs4 import BeautifulSoup
import pandas as pd
import requests
import os

def get_hathi_links(page, file_name, df):
    '''This function scrapes volume links from a Hathi Trust record page in case the collection making is not working.'''
    result = requests.get(page)
    ht_page = result.content
    soup = BeautifulSoup(ht_page, 'html.parser')
    if df:
        write_dataframe(soup, file_name)
    else:
        write_file(soup, file_name)
   

def write_dataframe(soup, output_path):
    links = soup.find_all(attrs={"data-hdl":True})

    for l in links:
        em = l.find_next_siblings()
        if 'Indiana' in em[0].get_text():
            vol_info = l.span.get_text().split(' ')
            print(vol_info)
            new_df = {}
            new_df['volume'] = ('_').join(vol_info[:-1]) 
            new_df['date'] = vol_info[-1]
            
            new_df['vol_id'] = l.get('data-hdl')
            print(new_df)
            dl = pd.DataFrame().append(new_df, ignore_index=True)
            if os.path.exists(output_path):
                dl.to_csv(output_path, mode='a', header=False, index=False)
            else:
                dl.to_csv(output_path, header=True, index=False)

def write_file(soup, file_name):
    f = open(file_name,'w')
    links = soup.find_all(attrs={"data-hdl":True})
    pd = DataFrame()
    for l in links:
        vol_id = l.get('data-hdl')
        
        if vol_id == 'uva.x004015666':
            break
        f.write(vol_id + '\n')

get_hathi_links('https://catalog.hathitrust.org/Record/003839852', '../data_sources/hathi_trust_metadatas/nashrat_akhbar_jamiat_al-Duwal_al-Arabiyah_1962_1967_003839852.csv', True)

'''
For Afro Asian Bulletin
em = l.find_next_siblings()
if 'Michigan' in em[0].get_text():
    continue
else:
    vol_info = l.span.get_text().replace(',', ' ').split(' ')
    print(vol_info)
    new_df = {}
    new_df['volume'] = ('_').join(vol_info[:-1]) 
    new_df['date'] = vol_info[-1]
    
    new_df['vol_id'] = l.get('data-hdl')
    print(new_df)
    dl = pd.DataFrame().append(new_df, ignore_index=True)
    if os.path.exists(output_path):
        dl.to_csv(output_path, mode='a', header=False, index=False)
    else:
        dl.to_csv(output_path, header=True, index=False)
'''

'''
For Mozambique Revolution 1971-1974
em = l.find_next_siblings()
    if 'Michigan' in em[0].get_text():
        vol_info = l.span.get_text().replace(',', ' ').split(' ')
        print(vol_info)
        new_df = {}
        new_df['volume'] = ('_').join(vol_info[:-1]) 
        new_df['date'] = vol_info[-1]
        
        new_df['vol_id'] = l.get('data-hdl')
        print(new_df)
        dl = pd.DataFrame().append(new_df, ignore_index=True)
        if os.path.exists(output_path):
            dl.to_csv(output_path, mode='a', header=False, index=False)
        else:
            dl.to_csv(output_path, header=True, index=False)
'''

'''
For MEN News Weekly
# # if l.get('data-hdl') == 'inu.30000122990637':
        #     # break ##Use this for '../data_sources/hathi_trust_metadatas/middle_east_news_economic_weekly_1964_1973_008564927.csv' -> 'uva.x004015666'
        #     ##Use this for '../data_sources/hathi_trust_metadatas/M.E.N_weekly_review_of_world_and_Arab_affairs.1963_1964_008564926.csv' -> 'inu.30000122990637'
'''