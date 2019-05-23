import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import glob
from progress.bar import IncrementalBar
import warnings
warnings.filterwarnings('ignore')

directory = '../data/'
for root, dirs, files in os.walk(directory):
        for file in files:
            if 'congo_corpus' in file:
                print(file)
                df = pd.read_csv(directory + file)
                print(df.columns)