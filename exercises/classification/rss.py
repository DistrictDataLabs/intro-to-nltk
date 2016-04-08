import os
import nltk
import time
import random
import pickle
import string

from bs4 import BeautifulSoup
from nltk.corpus import CategorizedPlaintextCorpusReader

# The first group captures the category folder, docs are any HTML file.
CORPUS_ROOT = './corpus'
DOC_PATTERN = r'(?!\.).*\.html'
CAT_PATTERN = r'([a-z_]+)/.*'

# Specialized Corpus Reader for HTML documents
class CategorizedHTMLCorpusreader(CategorizedPlaintextCorpusReader):
    """
    Reads only the HTML body for the words and strips any tags.
    """

    def _read_word_block(self, stream):
        soup = BeautifulSoup(stream, 'lxml')
        return self._word_tokenizer.tokenize(soup.get_text())

    def _read_sent_block(self, stream):
        sents = []
        soup  = BeautifulSoup(stream, 'lxml')
        piter = soup.find_all('p') if soup.find('p') else self._para_block_reader(stream)
        for para in piter:
            sents.extend([self._word_tokenizer.tokenize(sent)
                          for sent in self._sent_tokenizer.tokenize(para)])
        return sents

    def _read_para_block(self, stream):
        soup  = BeautifulSoup(stream, 'lxml')
        paras = []
        piter = soup.find_all('p') if soup.find('p') else self._para_block_reader(stream)

        for para in piter:
            paras.append([self._word_tokenizer.tokenize(sent)
                          for sent in self._sent_tokenizer.tokenize(para)])

        return paras

# Create our corpus reader
rss_corpus = CategorizedHTMLCorpusreader(CORPUS_ROOT, DOC_PATTERN,
                    cat_pattern=CAT_PATTERN, encoding='utf-8')

def timeit(func):
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        finit  = time.time()
        return result, finit-start
    return wrapper

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

def document_features(document):
    words = nltk.FreqDist(normalize_words(document))
    feats = {}
    for word in words.keys():
        feats['contains(%s)' % word] = True
    return feats

# Write our training, development, and test data sets.
@timeit
def generate_datasets(test_size=550, pickle_dir="."):
    """
    Creates three data sets; a test set and dev test set of 550 documents
    then a training set with the rest of the documents in the corpus. It
    will then write the data sets to disk at the pickle_dir.
    """
    documents = [(document_features(rss_corpus.words(fileid)), category)
                    for category in rss_corpus.categories()
                    for fileid in rss_corpus.fileids(category)]

    random.shuffle(documents)

    datasets = {
        'test':     documents[0:test_size],
        'devtest':  documents[test_size:test_size*2],
        'training': documents[test_size*2:],
    }

    for name, data in datasets.items():
        with open(os.path.join(pickle_dir, name+".pickle"), 'wb') as out:
            pickle.dump(data, out)

def load_datasets(pickle_dir="."):
    """
    Loads the randomly shuffled data sets from their pickles on disk.
    """

    def loader(name):
        path = os.path.join(pickle_dir, name+".pickle")
        with open(path, 'rb') as f:
            data = pickle.load(f)

        return name, data

    return dict(loader(name) for name in ('test', 'devtest', 'training'))

@timeit
def train_classifier(training, path='classifier.pickle'):
    """
    Trains the classifier and saves it to disk.
    """
    # classifier = nltk.MaxentClassifier.train(training,
    #             algorithm='megam', trace=2, gaussian_prior_sigma=1)
    classifier = nltk.NaiveBayesClassifier.train(training)

    with open(path, 'wb') as out:
        pickle.dump(classifier, out)

    return classifier

if __name__ == '__main__':
    # _, delta = generate_datasets(pickle_dir='datasets')
    # print "Took %0.3f seconds to generate datasets" % delta

    datasets = load_datasets(pickle_dir='datasets')
    classifier, delta = train_classifier(datasets['training'])
    print "trained in %0.3f seconds" % delta

    testacc    = nltk.classify.accuracy(classifier, datasets['test']) * 100
    devtestacc = nltk.classify.accuracy(classifier, datasets['devtest']) * 100

    print "test accuracy %0.2f%% -- devtest accuracy %0.2f%%" % (testacc, devtestacc)

    classifier.show_most_informative_features(30)

    # words = nltk.FreqDist(rss_corpus.words())
    # vocab = len(words.keys())
    # count = sum(words.values())
    # lexdi = float(count) / float(vocab)
    # print "Vocab: %i in %i words for a lexical diversity of %0.3f" % (vocab, count, lexdi)
    # print "%i files in %i categories" % (len(rss_corpus.fileids()), len(rss_corpus.categories()))
