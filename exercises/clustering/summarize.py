from __future__ import division

import math
import nltk
import datetime, re, sys
from random import randint

from sklearn.feature_extraction.text import TfidfVectorizer

reuters = nltk.corpus.reuters

lemmatizer = nltk.stem.WordNetLemmatizer()
def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text)
              for word in nltk.word_tokenize(sent)]

    return [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]



token_dict = {}
for article in reuters.fileids():
    token_dict[article] = reuters.raw(article)
tfidf = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words='english', decode_error='ignore')
print 'building term-document matrix... [process started: ' + str(datetime.datetime.now()) + ']'
# this can take some time (about 60 seconds on my machine)
tdm = tfidf.fit_transform(token_dict.values())
print 'done! [process finished: ' + str(datetime.datetime.now()) + ']'
feature_names = tfidf.get_feature_names()
print 'TDM contains ' + str(len(feature_names)) + ' terms and ' + str(tdm.shape[0]) + ' documents'
print 'first term: ' + feature_names[0]
print 'last term: ' + feature_names[len(feature_names)- 1]
for i in range(0, 4):
    print 'random term: ' + feature_names[randint(1,len(feature_names) - 2)]


article_id = randint(0, tdm.shape[0] - 1)
article_text = reuters.raw(reuters.fileids()[article_id])
sent_scores = []
for sentence in nltk.sent_tokenize(article_text):
    score = 0
    sent_tokens = tokenize_and_stem(sentence)
    for token in (t for t in sent_tokens if t in feature_names):
        score += tdm[article_id, feature_names.index(token)]
    sent_scores.append((score / len(sent_tokens), sentence))

summary_length = int(math.ceil(len(sent_scores) / 5))
sent_scores.sort(key=lambda sent: sent[0])
print '*** SUMMARY ***'
for summary_sentence in sent_scores[:summary_length]:
    print summary_sentence[1]
print '\n*** ORIGINAL ***'
print article_text
