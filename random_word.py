import urllib.request
import random

word_url = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
response = urllib.request.urlopen(word_url)
long_txt = response.read().decode()
words = long_txt.splitlines()

upper_words = [word for word in words if word[0].isupper()]
name_words  = [word for word in upper_words if not word.isupper()]

def get_random_name():
    return '_'.join([name_words[random.randint(0, len(name_words))] for i in range(2)])
