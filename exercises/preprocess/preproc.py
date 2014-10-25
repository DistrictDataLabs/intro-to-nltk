#!/usr/bin/env python

import os
import bs4
import nltk
import codecs
import string

def preprocess(path):
    """
    Opens a file, reads the contents - then performs the following tasks:

    1. Gets all the paragraphs in the text
    2. Segments the paragraphs with the sent_tokenizer
    3. Tokenizes the sentences with the word_tokenizer
    4. Tags the sentences using the default pos_tagger

    It then yields a list of paragraphs whose data structure is a list of
    sentences that are tokenized and tagged.
    """

    with open(path, 'r') as data:
        soup = bs4.BeautifulSoup(data)
        for tag in soup.find_all('p'):
            paragraph = tag.get_text()
            sentences = nltk.sent_tokenize(paragraph)
            for idx, sentence in enumerate(sentences):
                sentences[idx] = nltk.pos_tag(nltk.word_tokenize(sentence))
            yield sentences

def transform(htmldir, textdir):
    """
    Pass in a directory containing HTML documents and an output directory
    for the preprocessed text and this function transforms the HTML to a
    text corpus that has been tagged in the Brown corpus style.
    """
    for name in os.listdir(htmldir):
        inpath  = os.path.join(htmldir, name)
        outpath = os.path.join(textdir, os.path.splitext(name)[0] + ".txt")
        if os.path.isfile(inpath):
            with codecs.open(outpath, 'w+', encoding='utf-8') as f:
                for paragraph in preprocess(inpath):
                    for sentence in paragraph:
                        f.write(" ".join("%s/%s" % (word, tag) for word, tag in sentence))
                        f.write("\n")
                    f.write("\n")

def evaluation(textdir):
    """
    Uses the nltk.TaggedCorpusReader to answer the evaluation questions.
    """

    # Construct the corpus
    corpus    = nltk.corpus.TaggedCorpusReader(textdir, r'.*txt')

    # Construct stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(list(string.punctuation))          # Remove punctuation
    stopwords.extend(["''", '``', "'s", "n't", "'ll"])  # Custom stopwords

    # Get the interesting words from corpus
    words     = [word.lower() for word in corpus.words() if word not in stopwords]

    # Count the words and tags
    tokens    = nltk.FreqDist(corpus.words())
    unigrams  = nltk.FreqDist(words)
    bigrams   = nltk.FreqDist(nltk.bigrams(words))
    tags      = nltk.FreqDist(tag for word, tag in corpus.tagged_words())

    # Eliminate stopwords
    for word in stopwords:
        unigrams.pop(word, None)
        bigrams.pop(word, None)

    # Enumerate the vocabulary and word count
    vocab     = len(tokens)            # The number of unique tokens
    count     = sum(tokens.values())   # The word count for the entire corpus

    # Answer the evaluation questions
    print "This corpus contains %i words with a vocabulary of %i tokens."  % (count, vocab)
    print "The lexical diversity is %0.3f" % (float(count) / float(vocab))

    print "The 5 most common tags are:"
    for idx, tag in enumerate(tags.most_common(5)):
        print "    %i. %s (%i samples)" % ((idx+1,) + tag)

    print "\nThe 10 most common unigrams are:"
    for idx, tag in enumerate(unigrams.most_common(10)):
        print "    %i. %s (%i samples)" % ((idx+1,) + tag)

    print "\nThe 10 most common bigrams are:"
    for idx, tag in enumerate(bigrams.most_common(10)):
        print "    %i. %s (%i samples)" % ((idx+1,) + tag)

    print "\nThere are %i nouns in the corpus" % sum(val for key,val in tags.items() if key.startswith('N'))

if __name__ == '__main__':
    transform('corpus', 'newcorpus')
    # evaluation('newcorpus')
