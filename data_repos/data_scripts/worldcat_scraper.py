from bs4 import BeautifulSoup
import re
import requests
import pandas as pd
import os

for i in range(110, 630, 10):
    url = 'https://www.worldcat.org/search?q=au%3Agamal+abdel+nasser&fq=ap%3A%22nasser%2C+gamal+abdel%22&dblist=638&start={}&qt=page_number_link'.format(i)
    print(i)

    

    result = requests.get(url)
    page = result.content
    soup = BeautifulSoup(page, 'html.parser')
    citations = soup.find_all('td', class_='result details')
    for item in citations:
        title = item.find('div', class_='name').get_text()
        author = item.find('div', class_='author').get_text()
        publisher = item.find('div', class_='publisher')
        if publisher is not None:
            publisher = publisher.get_text().replace("]","").replace("[", "")
        else:
            publisher = 'No publisher'


        df = {'author_searched': "Gamal Abdel Nasser", 'url': url, 'title': title, 'author_listed': author, 'publisher': publisher}
    
        # print(df)
        page_df = pd.DataFrame(df, index=[0])
        output_path = 'nasser_citations_worldcat.csv'
        if os.path.exists(output_path):
            page_df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            page_df.to_csv(output_path, header=True, index=False)

