import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import os
# open file and read in lines
with open('AO_August_7_1960.txt', 'r') as file_in:
    raw_text = file_in.readlines()
aug_7_1960 = ''.join(raw_text)
type(aug_7_1960)
nlp_lg = spacy.load('en_core_web_lg')
doc_nlp_sm = nlp_lg(aug_7_1960)
doc_sm = [word for word in doc_nlp_sm if word.is_stop == False and word.is_punct == False and word.is_space == False and word.is_digit == False and word.like_num == False]
docNounVecs_fp = [word.vector for word in doc_sm]
docNounNormVecs_fp = [word.vector_norm for word in doc_sm]
docNounLabels_fp = [word.string.strip() for word in doc_sm]
lsa = TruncatedSVD(n_components=2, n_iter=10)
lsaOut = lsa.fit_transform(docNounVecs_fp)
print(lsaOut.shape)
xs, ys = lsaOut[:,0], lsaOut[:,1]
for i in range(len(xs)):
    print(docNounLabels_fp[i], xs[i], ys[i], docNounNormVecs_fp[i])
    plt.scatter(xs[i], ys[i])
    plt.annotate(docNounLabels_fp[i], (xs[i], ys[i]))
print('finished')
plt.show()
