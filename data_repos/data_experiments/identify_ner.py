import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import glob
from progress.bar import IncrementalBar
import warnings
warnings.filterwarnings('ignore')


def get_pages(input_file, output_path, terms):
    input_df = pd.read_csv(input_file)
    frames = []
    print(input_df.columns)
    ner_data = IncrementalBar('ner data for volume', max=len(input_df.index))
    for index, row in input_df.iterrows():
        ner_data.next()
        df = pd.DataFrame(input_df.iloc[index]).transpose()
        df.reset_index(drop=True, inplace=True)

        for t in terms:
            text = df.loc[df.tokenized_text.str.contains(t, regex=False) == True]
            print(text)
            if len(text) > 0:
                
                counts = df.tokenized_text.apply(lambda x: x.count(t))
                if int(counts) > 0:
                    
                    text['term'] = t
                    text['word_counts'] = counts
                    text.reset_index(drop=True, inplace=True)
                    frames.append(text)
    ner_data.finish()
    df = pd.concat(frames, ignore_index=True)

    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_path, header=True, index=False)

def get_hathi_pages(input_file, output_path, terms):
    input_df = pd.read_csv(input_file)
    frames = []
    print(input_df.columns)
    ner_data = IncrementalBar('ner data for volume', max=len(input_df.index))
    for index, row in input_df.iterrows():
        ner_data.next()
        df = pd.DataFrame(input_df.iloc[index]).transpose()
        df.reset_index(drop=True, inplace=True)

        for t in terms:
            text = df.loc[df.tokenized_text.str.contains(t, regex=False) == True]
            if len(text) > 0:
                
                counts = df.tokenized_text.apply(lambda x: x.count(t))
                if int(counts) > 0:
                    
                    text['term'] = t
                    text['word_counts'] = counts
                    text.reset_index(drop=True, inplace=True)
                    if os.path.exists(output_path):
                        text.to_csv(output_path, mode='a', header=False, index=False)
                    else:
                        text.to_csv(output_path, header=True, index=False)

    ner_data.finish()


    

if __name__ ==  "__main__" :
    terms = ['patrice','congo','lumumba','tshombe','leopoldville','belgian','mobutu','kasavubu','katanga']
    get_hathi_pages('../data/combined_all_data_1960_1966_binned_congo_classified_clean.csv', '../data/final_ner_congo_identified.csv', terms)

