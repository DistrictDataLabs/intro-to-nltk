#!/usr/bin/env python


import nltk

def tokenize(phrase):
    """
    Returns a grammar phrase based on the lexical input.
    """
    for token in nltk.word_tokenize(phrase):
        token = token.lower()
        yield token

if __name__ == '__main__':

    grammar = nltk.data.load('file:nounphrases.cfg')
    parser  = nltk.ChartParser(grammar)

    while True:
        try:
            phrase = tokenize(raw_input("Enter phrase to parse: "))
            for tree in parser.parse(phrase):
                print tree.pprint()
        except KeyboardInterrupt:
            break
