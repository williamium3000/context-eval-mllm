import itertools
from nltk.corpus import wordnet as wn


def get_synset(word):
    syn_list = wn.synonyms(word)
    syn_list = list(itertools.chain.from_iterable(syn_list))
    return [word]+[' '.join(syn.split('_')) for syn in syn_list]
