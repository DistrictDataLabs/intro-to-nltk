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

if __name__ == '__main__':
    transform('corpus', 'newcorpus')
