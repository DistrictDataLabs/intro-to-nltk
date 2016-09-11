#!/usr/bin/env python3

import os
import bs4
import nltk
import time
import pickle

from tqdm import tqdm
import os.path as path
import multiprocessing as mp


def preprocess(inpath, outdir):
    """
    Preprocesses the HTML file at the inpath and writes it out with the same
    name to the output directory. Returns the number of bytes it processed.
    """
    time.sleep(0.8)

    with open(inpath, 'r') as f:
        # Parse the HTML using BeautifulSoup
        soup = bs4.BeautifulSoup(f, 'lxml')

        # Process the document into paragraphs, sentences, tokens, tags.
        doc  = [
            [
                nltk.pos_tag(nltk.word_tokenize(sent))
                for sent in nltk.sent_tokenize(paragraph.get_text())
            ]
            for paragraph in soup.find_all('p')
        ]

    # Compute the path to write the pickle to.
    name, ext = os.path.splitext(os.path.basename(inpath))
    name += ".pickle"
    outpath = os.path.join(outdir, name)

    # Dump the pickle to the outpath.
    with open(outpath, 'wb') as f:
        pickle.dump(doc, f)

    return os.path.getsize(inpath)


def parallelize(source, target, tasks=mp.cpu_count()):
    """
    Preprocesses all the HTML files in source and writes them as pickle files
    to the target directory. Tracks progress using a progress bar by showing
    how many bytes are being processed.
    """

    # Phase 0: Get all the HTML paths in the source directory and prep target
    paths = [
        path.join(source, name) for name in os.listdir(source)
    ]

    if not os.path.exists(target):
        os.makedirs(target)

    # Phase 1: Get the total filesize from all HTML files
    total = sum(os.path.getsize(html) for html in paths)

    # Phase 2: Create the progress bar and a callback for results
    with tqdm(total=total, unit='Bytes') as pbar:
        on_result = lambda result: pbar.update(result)

        # Phase 3: Create the multiprocessing pool and initialize work.
        # Note: apply_async is required for progress tracking, map_async will
        # only call the callback once the entire job has been completed.
        pool = mp.Pool(processes=tasks)
        for inpath in paths:
            pool.apply_async(preprocess, (inpath, target), callback=on_result)

        # Phase 4: Close the pool and join
        pool.close()
        pool.join()


if __name__ == '__main__':
    CURDIR = path.dirname(__file__)
    CORPUS = path.join(CURDIR, "corpus")
    OUTDIR = path.join(CURDIR, "tagged")

    parallelize(CORPUS, OUTDIR, tasks=2)
