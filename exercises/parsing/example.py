
import nltk

grammar = nltk.grammar.CFG.fromstring("""

S -> NP
NP -> N N | ADJP NP | DET N
ADJP -> ADJ NP

DET -> 'an'
N -> 'airplane'
""")

parser = nltk.parse.ChartParser(grammar)

p = list(parser.parse(nltk.word_tokenize("an airplane")))

for a in p:
    a.pprint()
    # p[0].draw()
