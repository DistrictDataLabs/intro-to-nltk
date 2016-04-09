import numpy
import nltk
import string

from itertools import groupby
from operator import itemgetter
from nltk.decorators import memoize
from nltk.cluster import KMeansClusterer, euclidean_distance

## Constants
lemmatizer  = nltk.stem.WordNetLemmatizer()
stopwords   = set(nltk.corpus.stopwords.words('english'))
punctuation = string.punctuation

corpus      = nltk.corpus.TaggedCorpusReader("corpus", r'.*txt')
vocab       = list(set(corpus.words()))

K = 10

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

def get_cluster(k=K):
    cluster = KMeansClusterer(k, euclidean_distance)
    cluster.cluster([vectorspaced(corpus.words(fileid)) for fileid in corpus.fileids()])
    return cluster

if __name__ == "__main__":
    cluster = get_cluster()

    # Take a look at the cluster ids for the documents 
    groups  = [
        (cluster.classify(vectorspaced(corpus.words(fileid))), fileid)
        for fileid in corpus.fileids()
    ]

    groups.sort(key=itemgetter(0)) 
    for group, items in groupby(groups, key=itemgetter(0)):
        for item in items:
            print "{}: {}".format(*item)
        
