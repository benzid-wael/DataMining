# -*- coding: utf-8 -*-

"""
A summarizer based on Luhn's algorithm.
"""

import nltk
import numpy

import os
import nltk.data
from nltk.corpus import PlaintextCorpusReader


PACKAGE_PATH = os.path.dirname(__file__)
CLUSTER_THRESHOLD = 5  # Distance between words to consider


def get_tokenizer(language):
    """ Returns a tokenizer for the given language. """
    subpath = "data/tokenizers/punkt/%s.pickle" % language
    path = os.path.join(PACKAGE_PATH, subpath)
    return nltk.data.load(path)


def get_stopwords():
    """ Returns a corpus of stopwords for several languages. """
    subpath = "data/corpora/stopwords"
    path = os.path.join(PACKAGE_PATH, subpath)
    corpus = PlaintextCorpusReader(path, '.*')
    return corpus


def compute_sentences_score(sentences, important_words, lang):
    """
    Compute score of the given sentences according to the given important
    words.
    """
    scores = []
    sentence_idx = -1

    # tokenizer = get_tokenizer(lang)
    # words = [tokenizer.tokenize(s) for s in sentences]
    words = [nltk.tokenize.word_tokenize(s) for s in sentences]

    for s in words:
        sentence_idx += 1
        word_idx = []

        # For each word in the word list...
        for w in important_words:
            try:
                # Compute an index for where any important words occur in the
                # sentence.
                word_idx.append(s.index(w))
            except ValueError, e:
                # w not in this sentence
                pass

        word_idx.sort()
        # It is possible that some sentences may not contain any important words
        # at all.
        if len(word_idx)== 0:
            continue

        # Using the word index, compute clusters by using a max distance
        # threshold for any two consecutive words.
        clusters = []
        cluster = [word_idx[0]]
        i = 1
        while i < len(word_idx):
            if word_idx[i] - word_idx[i - 1] < CLUSTER_THRESHOLD:
                cluster.append(word_idx[i])
            else:
                clusters.append(cluster[:])
                cluster = [word_idx[i]]
            i += 1
        clusters.append(cluster)

        # Score each cluster. The max score for any given cluster is the score
        # for the sentence.
        max_cluster_score = 0
        for c in clusters:
            significant_words_in_cluster = len(c)
            total_words_in_cluster = c[-1] - c[0] + 1
            score = 1.0 * significant_words_in_cluster \
                * significant_words_in_cluster / total_words_in_cluster

            if score > max_cluster_score:
                max_cluster_score = score
        scores.append((sentence_idx, score))

    return scores
