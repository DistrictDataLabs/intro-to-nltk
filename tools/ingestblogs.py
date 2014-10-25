#!/usr/bin/env python

import os
import bs4
import codecs
import feedparser

def feedlist(path='feedly.opml'):
    with open(path, 'r') as feedly:
        soup = bs4.BeautifulSoup(feedly)
        for outline in soup.find_all("outline"):
            if 'xmlurl' in outline.attrs:
                yield outline.get('xmlurl')

def fetch(path='feedly.opml'):
    for url in feedlist(path):
        data = feedparser.parse(url)
        for item in data.entries:
            yield item

def write(dirname, item):
    parts = item.id.split("/")
    slug  = parts[-1] if parts[-1] else parts[-2]
    path = os.path.join(dirname, slug+".html")

    with codecs.open(path, 'w+', encoding='utf-8') as out:
        out.write("<head>\n\t<title>%s</title>\n</head>\n" % item.title)
        out.write("<body>\n<h1>%s</h1>\n\n" % item.title)
        out.write(item.description)
        out.write("</body>")

if __name__ == '__main__':
    for item in fetch():
        write("corpus", item)
