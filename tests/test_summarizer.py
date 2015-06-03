# -*- coding: utf-8 -*-

from datamining.summarizer import Summarizer


class TestSummarizer:

    def test_one_sentence(self):
        summarizer = Summarizer()
        res = summarizer.summarize("Hello world")
        assert len(res) == 1

    def test_two_sentences(self):
        summarizer = Summarizer()
        txt = "My name is John Doe. I'm a software engineer."
        res = summarizer.summarize(txt)
        assert len(res) == 2

    def test_three_sentences(self):
        txt = "wa s s s wa s s s wa." + \
              "wb s wb s wb s s s s s s s s s wb." + \
              "wc s s wc s s wc"
        summarizer = Summarizer()
        res = summarizer.summarize(txt)
        assert len(res) == 1

    def test_short_scientific_text(self):
        txt = """
        Detecting patterns is a central part of Natural Language Processing.
        Words ending in -ed tend to be past tense verbs (5.). Frequent use of
        will is indicative of news text (3). These observable patterns - word
        structure and word frequency - happen to correlate with particular
        aspects of meaning, such as tense and topic. But how did we know where
        to start looking, which aspects of form to associate with which aspects
        of meaning?"""
        nbr_sentences = len(txt.split('.'))
        summarizer = Summarizer()
        res = summarizer.summarize(txt)
        # TODO Should we compute the rate of our summarization algorithm
        assert len(res) < nbr_sentences
