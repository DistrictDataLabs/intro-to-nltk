import nltk
import codecs
import requests
from readability.readability import Document
from bs4 import BeautifulSoup

def fetch(url, path):
    """
    Fetches a URL and saves it to disk at the path.
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as data:
            for chunk in response.iter_content(4096):
                data.write(chunk)
        return True
    return False

def textualize(path):
    """
    Opens an HTML file on disk and cleans up the tags to get the text
    """
    with codecs.open(path, 'r', 'utf8') as f:
        html = f.read()
        article = Document(html).summary()
        title   = Document(html).title()
        soup    = BeautifulSoup(article)

        return title, soup.text

def test():
    """
    Works on a fixed URL
    """
    url = "http://venturebeat.com/2014/07/04/facebooks-little-social-experiment-got-you-bummed-out-get-over-it/"
    fetch(url, "facebooks-little-social-experiment.html")

    title, content = textualize("facebooks-little-social-experiment.html")

    print title
    print
    print content

if __name__ == '__main__':
    import sys
    while True:
        try:
            url = raw_input("Enter URL: ")
            fetch(url, "document.html")
            title, content = textualize("document.html")
            print title
            print
            print content
        except KeyboardInterrupt:
            print
            sys.stdout.flush()
            break

