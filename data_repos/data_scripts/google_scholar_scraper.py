from bs4 import BeautifulSoup
import re
import requests
import pandas as pd
import os

for i in range(0, 100, 10):
    url = 'https://scholar.google.com/scholar?start={}&q=author:%22gamal+abdel+nasser%22&hl=en&as_sdt=0,7'.format(i)
    print(url, i)

    

    result = requests.get(url)
    page = result.content
    soup = BeautifulSoup(page, 'html.parser')
    citations = soup.find_all('div', class_='gs_r gs_or gs_scl')
    # citations = soup.find_all('div', class_='gs_a')
    # titles = soup.find_all('h3', class_='gs_rt')
    # all_titles=[]
    # for title in titles:

    #     all_titles.append(title.get_text().split(']')[-1])
    for index, link in enumerate(citations):
        title = link.find('h3', class_='gs_rt').get_text().split(']')[-1]
        df = {'author_searched': "Gamal Abdel Nasser", 'url': url, 'title': title}

        strings = link.find('div', class_='gs_a').get_text().split(' ')
        df['info'] = link.find('h3', class_='gs_rt').get_text()

        cited_by = link.find('div', class_='gs_fl').get_text().split(' ')
        for item in cited_by:
            try:
                int(item)
                df['cited'] = int(item)
                print('digit', item)
                break
            except ValueError:
                df['cited'] = 0
                print('not digit')
            

        
        for item in strings:
            try:
                int(item)
                df['year'] = int(item)
                print('digit', item)
                break
            except ValueError:
            
                print('not digit', item)
        print(df)
        
        page_df = pd.DataFrame(df, index=[0])
        output_path = 'nasser_citations_googlescholar.csv'
        if os.path.exists(output_path):
            page_df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            page_df.to_csv(output_path, header=True, index=False)

