from bs4 import BeautifulSoup
import requests

def get_hathi_links(page, file_name):
    '''This function scrapes volume links from a Hathi Trust record page in case the collection making is not working.'''
    result = requests.get(page)
    ht_page = result.content
    soup = BeautifulSoup(ht_page, 'html.parser')

    f = open(file_name,'w')
    links = soup.find_all(attrs={"data-hdl":True})
    for l in links:
        vol_id = l.get('data-hdl')
        f.write(vol_id + '\n')

get_hathi_links('https://catalog.hathitrust.org/Record/008564927', '../data_sources/hathi_trust_metadatas/Middle_East_News_economic weekly_008564927.txt')