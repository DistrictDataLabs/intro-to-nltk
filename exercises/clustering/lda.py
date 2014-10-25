import gensim

import nltk
import string
from nltk.decorators import memoize

## Constants
lemmatizer  = nltk.stem.WordNetLemmatizer()
stopwords   = set(nltk.corpus.stopwords.words('english'))
punctuation = string.punctuation

@memoize
def normalize(token):
    token = token.lower()
    token = lemmatizer.lemmatize(token)
    return token

def normalize_words(words):
    for token in words:
        token = normalize(token)
        if token not in stopwords and token not in punctuation:
            yield token

def documents(path="corpus"):
    corpus = nltk.corpus.TaggedCorpusReader(path, r'.*txt')
    for fileid in corpus.fileids():
        yield list(normalize_words(corpus.words(fileid)))

def token_features(words):
    for token in words:
        token = normalize(token)
        if token not in stopwords and token not in punctuation:
            yield token


if __name__ == '__main__':
    id2word = gensim.corpora.Dictionary(documents())
    # id2word = gensim.corpora.Dictionary.load('corpus.txt')
    corpus  = [id2word.doc2bow(doc) for doc in documents()]
    # gensim.corpora.MmCorpus.serialize('corpus.mm', corpus)

    mm = gensim.corpora.MmCorpus('corpus.mm')

    print mm
    lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=10, update_every=1, passes=20)
    print lda.print_topics(10)
