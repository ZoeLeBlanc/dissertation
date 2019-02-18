import re
import sys
import textract
import PyPDF2

def parse_pdf(file_path):
    '''Method to parse a pdf and turn it into a pandas data table with text from each page'''
    pdf = PyPDF2.PdfFileReader(open(file_path, "rb"))
    num_of_pages = pdf.getNumPages()
    df_text_page = pd.DataFrame(columns=['page_number', 'text'])
    for i in range(num_of_pages):
        page = pdf.getPage(i)
        text = page.extractText()
        df_text_page.loc[len(df_text_page)] = [i,text]
