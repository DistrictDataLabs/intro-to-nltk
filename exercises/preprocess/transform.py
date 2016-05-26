#!/usr/bin/env python

import os
import bs4
import nltk
import codecs
import string

from readability.readability import Document

# Tags to extract as paragraphs from the HTML text
TAGS = [
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'p', 'li'
]

def preprocess(path):
    """
    Opens a file, reads the contents - then performs the following tasks:

    1. Summarize the text with readability
    1. Gets all the target tags in the text
    2. Segments the paragraphs with the sent_tokenizer
    3. Tokenizes the sentences with the word_tokenizer
    4. Tags the sentences using the default pos_tagger

    It then yields a list of paragraphs whose data structure is a list of
    sentences that are tokenized and tagged.
    """

    with open(path, 'r') as f:

        # Transform the document into a readability paper summary
        html = Document(f.read()).summary()

        # Parse the HTML using BeautifulSoup
        soup = bs4.BeautifulSoup(html)

        # Extract the paragraph delimiting elements
        for tag in soup.find_all(TAGS):

            # Get the HTML node text
            paragraph = tag.get_text()

            # Sentence Tokenize
            sentences = nltk.sent_tokenize(paragraph)
            for idx, sentence in enumerate(sentences):
                # Word Tokenize and Part of Speech Tagging
                sentences[idx] = nltk.pos_tag(nltk.word_tokenize(sentence))

            # Yield a list of sentences (the paragraph); each sentence of
            # which is a list of tuples in the form (token, tag).
            yield sentences


def transform(htmldir, textdir):
    """
    Pass in a directory containing HTML documents and an output directory
    for the preprocessed text and this function transforms the HTML to a
    text corpus that has been tagged in the Brown corpus style.
    """
    # List the target HTML directory
    for name in os.listdir(htmldir):

        # Determine the path of the file to transform and the file to write to
        inpath  = os.path.join(htmldir, name)
        outpath = os.path.join(textdir, os.path.splitext(name)[0] + ".txt")

        # Open the file for reading UTF-8
        if os.path.isfile(inpath):
            with codecs.open(outpath, 'w+', encoding='utf-8') as f:

                # Write paragraphs double newline separated and sentences
                # separated by a single newline. Also write token/tag pairs.
                for paragraph in preprocess(inpath):
                    for sentence in paragraph:
                        f.write(" ".join("%s/%s" % (word, tag) for word, tag in sentence))
                        f.write("\n")
                    f.write("\n")


if __name__ == '__main__':
    transform('corpus', 'newcorpus')
