# -*- coding: utf-8 -*-

"""
A summarizer based on Luhn's algorithm.
"""

import nltk
import numpy
import math

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


def summarize(txt, lang=None, N=None):
    """ Return a summary of the given text `txt`. """
    lang = lang or 'english'
    # number of words to consider important
    # XXX As this will be used mainly for micro-blogs and blogs it should not be
    # a big integer
    N = N or 10

    tokenizer = get_tokenizer(lang)
    sentences = [s for s in tokenizer.tokenize(txt)]

    normalized_sentences = [s.lower() for s in sentences]

    words = [w for sentence in normalized_sentences for w in
             tokenizer.tokenize(sentence)]

    # Compute frequency distribution
    fdist = nltk.FreqDist(words)

    stopwords_corpus = get_stopwords()
    top_n_words = [w[0] for w in fdist.items()
                   if w[0] not in stopwords_corpus.words(lang)][:N]

    scored_sentences = compute_sentences_score(normalized_sentences,
                                               top_n_words, lang)

    # Filter out nonsignificant sentences by using the average score plus a
    # fraction of the std dev as a filter
    avg = numpy.mean([s[1] for s in scored_sentences])
    std = numpy.std([s[1] for s in scored_sentences])
    threshold = avg + 0.5 * std
    mean_scored = [(sent_idx, score) for (sent_idx, score) in scored_sentences
                   if score >= threshold]

    return [sentences[idx] for (idx, score) in mean_scored]


class Summarizer(object):

    def __init__(self, lang=None, max_significat_words=None):
        self._max_significat_words = max_significat_words or 10
        self._default_language = lang or 'english'
        # FIXME We should find a proper way to install all needed data when
        # installing the package

    def summarize(self, txt):
        sentences = [s for s in nltk.tokenize.sent_tokenize(txt.strip())]

        normalized_sentences = [s.lower() for s in sentences]

        words = [w for sentence in normalized_sentences for w in
                 nltk.tokenize.word_tokenize(sentence)]

        # Compute frequency distribution
        fdist = nltk.FreqDist(words)

        top_n_words = [w[0] for w in fdist.items() if w[0] not
                       in nltk.corpus.stopwords.words(self._default_language)
                      ][:self._max_significat_words]

        scored_sentences = compute_sentences_score(normalized_sentences,
                                                   top_n_words,
                                                   self._default_language)

        # Filter out nonsignificant sentences by using the average score plus a
        # fraction of the std dev as a filter
        avg = numpy.mean([s[1] for s in scored_sentences])
        std = numpy.std([s[1] for s in scored_sentences])
        threshold = avg + 0.5 * std
        mean_scored = [(sent_idx, score) for (sent_idx, score) in scored_sentences
                       if score >= threshold]

        # XXX this heuristic should be improved
        if len(mean_scored) < 3:
            # NOTE At this stage, we should choice another strategy to summarize
            # the given text.
            top_sentences = max(3, int(math.sqrt(len(scored_sentences))))
            return [sentences[idx] for (idx, score) in scored_sentences]

        return [sentences[idx] for (idx, score) in mean_scored]
