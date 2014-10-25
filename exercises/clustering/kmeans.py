import numpy
import nltk
import string

from nltk.decorators import memoize
from nltk.cluster import KMeansClusterer, euclidean_distance

## Constants
lemmatizer  = nltk.stem.WordNetLemmatizer()
stopwords   = set(nltk.corpus.stopwords.words('english'))
punctuation = string.punctuation

corpus      = nltk.corpus.TaggedCorpusReader("corpus", r'.*txt')
vocab       = list(set(corpus.words()))

@memoize
def normalize(token):
    token = token.lower()
    token = lemmatizer.lemmatize(token)
    return token

def token_features(words):
    for token in words:
        token = normalize(token)
        if token not in stopwords and token not in punctuation:
            yield token

def vectorspaced(words):
    features = list(token_features(words))
    return numpy.array([token in features for token in vocab], numpy.short)

def get_cluster():
    cluster = KMeansClusterer(10, euclidean_distance)
    cluster.cluster([vectorspaced(corpus.words(fileid)) for fileid in corpus.fileids()])
    return cluster
