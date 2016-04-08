#!/usr/bin/env python
import sys
import pickle

from products import product_features

MAXENT = 'products_maxent.pickle'
BAYES  = 'products_bayes.pickle'

if __name__ == '__main__':
    product = product_features({
        'name': sys.argv[1],
        'description': sys.argv[2],
    })

    with open(MAXENT, 'r') as f:
        classifier = pickle.load(f)
        prediction = classifier.classify(product)

        print "Product {!r} is classified as {!r}\n\n".format(
            sys.argv[1], prediction)

        classifier.explain(product)
