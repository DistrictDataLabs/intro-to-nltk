import nltk
import time
import string
import random
import pickle
import unicodecsv as csv

def timeit(func):
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        finit  = time.time()
        return result, finit-start
    return wrapper

class ProductCorpus(object):
    """
    Simple product "CorpusReader" that implements a CorpusReader-like
    interface including products, words, categories, etc.
    """

    def __init__(self, path):
        self.path = path
        self._categories = None

    def __iter__(self):
        """
        Iterates through each product in the dataset
        """
        with open(self.path, 'r') as data:
            reader = csv.DictReader(data)
            for idx, row in enumerate(reader):
                row['rowid'] = idx + 1
                yield row

    def categories(self):
        if not self._categories:
            self._categories = set(row['category'] for row in self)
        return self._categories

    def products(self, categories=None):
        if not categories:
            categories = self.categories()
        elif isinstance(categories, basestring):
            categories = [categories]

        return filter(lambda x: x['category'] in categories, self)

# Create feature extractor methodology
def normalize_words(document):
    """
    Expects as input a list of words that make up a document. This will
    yield only lowercase significant words (excluding stopwords and
    punctuation) and will lemmatize all words to ensure that we have word
    forms that are standardized.
    """
    stopwords  = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    for token in document:
        token = token.lower()
        if token in string.punctuation: continue
        if token in stopwords: continue
        yield lemmatizer.lemmatize(token)

def product_features(product):
    name = nltk.FreqDist(normalize_words(nltk.wordpunct_tokenize(product['name'])))
    desc = nltk.FreqDist(normalize_words(nltk.wordpunct_tokenize(product['description'])))
    feats = {}
    for word in name.keys():
        feats['name(%s)' % word] = True

    for word in desc.keys():
        feats['description(%s)' % word] = True
    return feats

def generate_datasets(path):
    """
    Returns 10 percent of the corpus as test set and the rest for training.
    """

    corpus = ProductCorpus(path)
    prods  = [(product_features(product), product['category'])
                for product in corpus.products()]
    offset = len(prods)/10
    random.shuffle(prods)

    return prods[:offset], prods[offset:]

@timeit
def build_maxent(training, outpath):
    classifier = nltk.MaxentClassifier.train(training,
                    algorithm='megam', trace=2, gaussian_prior_sigma=1)

    with open(outpath, 'wb') as out:
        pickle.dump(classifier, out)

    return classifier

@timeit
def build_bayes(training, outpath):
    classifier = nltk.NaiveBayesClassifier.train(training)

    with open(outpath, 'wb') as out:
        pickle.dump(classifier, out)

    return classifier

@timeit
def build_dtree(training, outpath):
    classifier = nltk.DecisionTreeClassifier.train(training)

    with open(outpath, 'wb') as out:
        pickle.dump(classifier, out)

    return classifier

if __name__ == '__main__':

    train, test   = generate_datasets('products.csv')
    maxent, mdelta = build_maxent(train, 'products_maxent.pickle')
    bayes,  bdelta = build_bayes(train, 'products_bayes.pickle')
    # dtree,  ddelta = build_dtree(train, 'products_dtree.pickle')

    print "Maximum Entropy took %0.3f seconds to train" % mdelta
    print "    Accuracy %0.3f%%" % (nltk.classify.accuracy(maxent, test) * 100)
    print

    print "Naive Bayes took %0.3f seconds to train" % bdelta
    print "    Accuracy %0.3f%%" % (nltk.classify.accuracy(bayes, test) * 100)
    print

    # print "Decision Tree took %0.3f seconds to train" % ddelta
    # print "    Accuracy %0.3f%%" % (nltk.classify.accuracy(dtree, test) * 100)
    # print
