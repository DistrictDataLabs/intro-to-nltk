import nltk
import string

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
    evaluation('newcorpus')
