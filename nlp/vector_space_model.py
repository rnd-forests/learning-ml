import nltk
import re
import glob
import string
import math
from nltk import bigrams, trigrams
from nltk.tokenize import RegexpTokenizer

files = glob.glob("./data/*.txt")
stopwords = nltk.corpus.stopwords.words('english')
tokenizer = RegexpTokenizer("[\w']+", flags=re.UNICODE)


def freq(word, doc):
    return doc.count(word)


def word_count(doc):
    return len(doc)


def tf(word, doc):
    return freq(word, doc) / float(word_count(doc))


def num_docs_containing(word, list_of_docs):
    count = 0
    for document  in list_of_docs:
        if (freq(word, document)) > 0:
            count += 1
    return count + 1


def idf(word, list_of_docs):
    return math.log(len(list_of_docs) / float(num_docs_containing(word, list_of_docs)))


def get_tokens():
    tokens = []
    for file in files:
        with open(file) as reader:
            contents = reader.read().translate(string.punctuation)
            tokens.extend(tokenizer.tokenize(contents))
    return tokens


vocabulary = []
all_tips = []
tokens = get_tokens()

bitokens = bigrams(tokens)
tritokens = trigrams(tokens)
tokens = [token.lower() for token in tokens if len(token) > 2]
tokens = [token for token in tokens if token not in stopwords]

bitokens = [' '.join(token).lower() for token in bitokens]
bitokens = [token for token in bitokens if token not in stopwords]

tritokens = [' '.join(token).lower() for token in tritokens]
tritokens = [token for token in tritokens if token not in stopwords]

ftokens = []
ftokens.extend(tokens)
ftokens.extend(bitokens)
ftokens.extend(tritokens)

docs = {'freq': {}, 'tf': {}, 'idf': {}}
for token in ftokens:
    docs['freq'][token] = freq(token, ftokens)
    docs['tf'][token] = tf(token, ftokens)

vocabulary.append(ftokens)
for doc in docs:
    for token in docs[doc]['tf']:
        docs[doc]['idf'][token] = idf(token, vocabulary)

print(docs)
