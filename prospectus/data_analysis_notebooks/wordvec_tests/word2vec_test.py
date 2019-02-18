import gensim
import nltk
nltk.download('punkt')
import string
import csv
from nltk.corpus import stopwords
nltk.download('stopwords')


# open file and read in lines
with open('ao_output.txt', 'r') as file_in:
    raw_text = file_in.readlines()

# take the text file and get sentences.

sentences = []
for line in raw_text:
    sentences.extend(nltk.sent_tokenize(line))

tokenized_sentences = []
for sentence in sentences:
    # tokenize each sentenceand lowercase all the tokens
    sent_tokens = [token.lower() for token in nltk.word_tokenize(sentence) if token not in string.punctuation and token not in stopwords.words('english')]
    # add the list of lowercased tokens to tokenized_sentences
    tokenized_sentences.append(sent_tokens)

model =  gensim.models.Word2Vec(       tokenized_sentences, size=100, window=5, sg=0, alpha=0.025,iter=5, max_vocab_size=None)

# model.wv.most_similar(positive=['zionist'], negative=['arab'])
