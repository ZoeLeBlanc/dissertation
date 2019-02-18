import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import os


df=pd.read_csv('full_page_image_lucida_test.csv')
sp_df = pd.read_csv('split_page_image_lucida_test.csv', encoding='utf-8')
# sp_df = pd.read_csv('Arab_Observer_Split_Page_1960_issues.csv', encoding='utf-8')
order_df = pd.read_csv('ordered_text_image_lucida_test.csv', encoding="ISO-8859-1")
order_df.fillna(0, inplace=True)

new_df = order_df.groupby(['page'])['google_vision_text'].apply(','.join).reset_index()

order_text = ''.join(new_df.google_vision_text)
f = open('order_output.txt', 'wt', encoding='utf-8')
f.write(order_text)

fp_text = ''.join(df.google_vision_text)
f = open('output.txt', 'wt', encoding='utf-8')
f.write(fp_text)

sp_text = ''.join(sp_df.google_vision_text)
f = open('output.txt', 'wt', encoding='utf-8')
f.write(sp_text)

nlp = spacy.load('en_core_web_lg')

doc_fp = nlp(fp_text)
doc_sp = nlp(sp_text)
doc_order = nlp(order_text)

wordVecs_fp = []
for word in doc_fp:
    if word.text == 'colonialism':
        for word2 in doc_fp:
            if word2.is_stop == False:
                if word.similarity(word2) > 0.3:
                    print(word.similarity(word2), word2.text, word2.has_vector, word2.vector_norm, word2.is_oov)
                    wordVecs_fp.append(word2)

wordVecs_sp = []
for word in doc_sp:
    if word.text == 'colonialism':
        for word2 in doc_sp:
            if word2.is_stop == False:
                if word.similarity(word2) > 0.3:
                    print(word.similarity(word2), word2.text, word2.has_vector, word2.vector_norm, word2.is_oov)
                    wordVecs_sp.append(word2)

wordVecs_order = []
for word in doc_order:
    if word.text == 'colonialism':
        for word2 in doc_order:
            if word2.is_stop == False:
                if word.similarity(word2) > 0.3:
                    print(word.similarity(word2), word2.text, word2.has_vector, word2.vector_norm, word2.is_oov)
                    wordVecs_order.append(word2)

# docNounVecs_fp = [word.vector for word in wordVecs_fp]
# docNounLabels_fp = [word.string.strip() for word in wordVecs_fp]
# lsa = TruncatedSVD(n_components=2, n_iter=10)
# lsaOut = lsa.fit_transform(docNounVecs_fp)
# lsaOut.shape
# %matplotlib notebook
# xs, ys = lsaOut[:,0], lsaOut[:,1]
# for i in range(len(xs)):
#     plt.scatter(xs[i], ys[i])
#     plt.annotate(docNounLabels_fp[i], (xs[i], ys[i]))
#
#
# docNounVecs_sp = [word.vector for word in wordVecs_sp]
# docNounLabels_sp = [word.string.strip() for word in wordVecs_sp]
# lsa = TruncatedSVD(n_components=2, n_iter=10)
# lsaOut = lsa.fit_transform(docNounVecs_sp)
# lsaOut.shape
# %matplotlib notebook
# xs, ys = lsaOut[:,0], lsaOut[:,1]
# for i in range(len(xs)):
#     plt.scatter(xs[i], ys[i])
#     plt.annotate(docNounLabels_sp[i], (xs[i], ys[i]))

docNounVecs_order = [word.vector for word in wordVecs_order]
docNounLabels_order = [word.string.strip() for word in wordVecs_order]
lsa = TruncatedSVD(n_components=2, n_iter=10)
lsaOut = lsa.fit_transform(docNounVecs_order)
lsaOut.shape
xs, ys = lsaOut[:,0], lsaOut[:,1]
for i in range(len(xs)):
    plt.scatter(xs[i], ys[i])
    plt.annotate(docNounLabels_order[i], (xs[i], ys[i]))

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(docNounVecs_order)
xs, ys = X_tsne[:,0], X_tsne[:,1]
for i in range(len(xs)):
    plt.scatter(xs[i], ys[i])
    plt.annotate(docNounLabels_order[i], (xs[i], ys[i]))
