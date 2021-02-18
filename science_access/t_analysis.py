# Scientific readability project
# authors ...,
# Russell Jarvis
# https://github.com/russelljjarvis/
# russelljarvis@protonmail.com
# Patrick McGurrin
# patrick.mcgurrin@gmail.com


import base64
import copy
import math
import os
import pickle
import re
import sys
import time
import collections

# import matplotlib  # Its not that this file is responsible for doing plotting, but it calls many modules that are, such that it needs to pre-empt
# matplotlib.use('Agg')

import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import cmudict, stopwords, subjectivity
from nltk.probability import FreqDist
from nltk.sentiment import SentimentAnalyzer
from nltk.tag.perceptron import PerceptronTagger
import nltk
from nltk.corpus import words as english_words
from nltk.tokenize import word_tokenize

from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd
import re
import streamlit as st
from .utils import black_string, clue_links, clue_words, comp_ratio, publication_check


# english_check
# from tabulate import tabulate
from textblob import TextBlob
from textstat.textstat import textstat

from science_access.utils import check_passive

tagger = PerceptronTagger(load=False)
# import matplotlib.pyplot as plt

# from science_access.enter_author_name import create_giant_strings#, not_want_list
not_want_list = [
    "article",
    "articlepubmedpubmed",
    "et",
    "al",
    "text",
    "crossref",
    "isigoogle",
    "cross",
    "ref",
    "google",
    "scholar",
    "article",
    "pubmed",
    "full",
    "doi",
    "org",
    "http",
    "copyright",
    "org",
    "figure",
    "pubmed",
    "accessshoping",
    "articlepubmedpubmed",
]


def create_giant_strings(ar, not_want_list):
    sci_corpus = ""
    first_pass = []
    for t in ar:
        if "tokens" in t.keys():
            for s in t["tokens"]:
                if s not in not_want_list:
                    first_pass.append(s)
    first_pass = set(first_pass)
    for s in first_pass:
        if "/" in s:
            temp = s.split("/")  # , " ")
            sci_corpus += str(" ") + temp[0]
            sci_corpus += str(" ") + temp[1]
        if "." in s:
            temp = s.split(".")  # , " ")
            sci_corpus += str(" ") + temp[0]
            sci_corpus += str(" ") + temp[1]
        if s not in set(not_want_list):
            sci_corpus += str(" ") + s  # +str(' ')
    return sci_corpus


def check_if_real_word(w):
    if w in english_words.words():
        return w
    else:
        return False


def unigram_zipf(tokens):
    """
    Get the zipf slope histogram for a corpus
    """
    model = collections.defaultdict(lambda: 0.01)
    tokens = [term for t in tokens for term in t]
    model = {}

    for word in tokens:
        count = model.get(word, 0)
        model[word] = count + 1
    """
    normalize observations relative to number of words in the model
    """
    for word in model:
        model[word] = model[word] / float(sum(model.values()))
    return model


#    https://github.com/nltk/nltk/blob/model/nltk/model/ngram.py


def entropy(self, text):
    """
    https://github.com/nltk/nltk/blob/model/nltk/model/ngram.py
    Calculate the approximate cross-entropy of the n-gram model for a
    given evaluation text.
    This is the average log probability of each word in the text.
    :param text: words to use for evaluation
    :type text: Iterable[str]
    """

    normed_text = (self._check_against_vocab(word) for word in text)
    H = 0.0  # entropy is conventionally denoted by "H"
    processed_ngrams = 0
    for ngram in self.ngram_counter.to_ngrams(normed_text):
        context, word = tuple(ngram[:-1]), ngram[-1]
        H += self.logscore(word, context)
        processed_ngrams += 1
    return -(H / processed_ngrams)


def perplexity(self, text):
    """
    Calculates the perplexity of the given text.
    This is simply 2 ** cross-entropy for the text.
    :param text: words to calculate perplexity of
    :type text: Iterable[str]
    """

    return pow(2.0, self.entropy(text))


def perplexity(testset, model):
    # https://stackoverflow.com/questions/33266956/nltk-package-to-estimate-the-unigram-perplexity
    perplexity = 1
    N = 0
    for word in testset:
        N += 1
        perplexity = perplexity + (1.0 / model[word])
    return perplexity


def bi_log_value(value):
    # Bi-symmetric log-like transformation, from:
    # http://iopscience.iop.org/article/10.1088/0957-0233/24/2/027001/pdf
    trans = np.sign(value) * np.log(1 + np.abs(value * 2.302585))
    return trans
    # df[col] = trans


# from spacy_langdetect import LanguageDetector
# import spacy
# nlp = spacy.load('en',disable=["parser"])
# nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
# nlp = spacy.load("en_core_web_sm")
# from spacy.lang.en import English
# nlp = English()#.from_disk("/model", disable=["parser"])
# tokenizer = nlp.Defaults.create_tokenizer(nlp)

# nlp = spacy.load("en_core_web_sm", disable=["parser"])
# nlp = English().from_disk("/model", disable=["parser"])
# doc = nlp("I don't want parsed", disable=["parser"])
"""
Use spacey to sense english faster
if doc._.language_scores['en'] <0.5:
    st.text('mangled_decoding')
    st.text(doc._.language)

    urlDat['mangled_decoding'] = True
    return urlDat
"""


ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words("english"))
NON_ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words()) - ENGLISH_STOPWORDS

STOPWORDS_DICT = {
    lang: set(nltk.corpus.stopwords.words(lang))
    for lang in nltk.corpus.stopwords.fileids()
}


def get_language(text):
    words = set(nltk.wordpunct_tokenize(text.lower()))
    return max(
        ((lang, len(words & stopwords)) for lang, stopwords in STOPWORDS_DICT.items()),
        key=lambda x: x[1],
    )[0]


def is_english(text):
    text = text.lower()
    words = set(nltk.wordpunct_tokenize(text))
    return len(words & ENGLISH_STOPWORDS) > len(words & NON_ENGLISH_STOPWORDS)


ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words("english"))


def complexityAlongtheText(text, chunk_length=128):
    words = text.split()
    cur = 0
    stds = []
    while cur < len(words):
        sub = words[cur : cur + chunk_length]
        sub_text = " ".join(sub)
        std = textstat.text_standard(sub_text, float_output=True)
        cur += chunk_length
        stds.append(std)
    return np.mean(stds), textstat.text_standard(text, float_output=True)


def text_proc(corpus, urlDat={}, WORD_LIM=50):

    if type(corpus) is type(str()):  # and not str("privacy policy") in corpus:
        corpus = corpus.replace("-", " ")  # remove characters that nltk can't read
        corpus = corpus.replace("/", " ")  # remove characters that nltk can't read
        corpus = corpus.replace(".", " ")  # remove characters that nltk can't read
        corpus = re.sub(
            r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(/\S+)?|\S+\.com\S+", " ", corpus
        )
        corpus = "".join([i for i in corpus if not i.isdigit()])

        if "Abstract" in corpus:
            corpus = corpus.split("Abstract")[1]

        elif "ABSTRACT" in corpus:
            corpus = corpus.split("ABSTRACT")[1]
        """
        if not "ABSTRACT" in corpus or "Abstract" in corpus:
            test = textstat.text_standard(corpus, float_output=True)
            if test > 90:
                print('premature end')
                print('premature end too much complexity \n\n\n\n\n\n\n\n')

                urlDat["page full of links"] = True
                return urlDat
        """
        tokens = word_tokenize(corpus)

        # tokens = tokenizer(corpus, disable=["parser"])
        # (len(tokens))

        stop_words = stopwords.words("english")
        # stop_words = (lex for lex in nlp.vocab if lex.is_stop)

        tokens = [word for word in tokens if not word in stop_words]
        tokens = [w.lower() for w in tokens if w.isalpha()]

        tokens = [w.lower() for w in tokens]  # make everything lower case
        """
        if not is_english(corpus):
            urlDat["mangled_decoding"] = True
            print('premature end not english \n\n\n\n\n')

            return urlDat
        """
        tokens = list(set(tokens) - set(not_want_list))
        # s.difference(t) s - t
        # new set with elements in s but not in t
        urlDat["wcount"] = textstat.lexicon_count(str(tokens))
        word_lim = bool(urlDat["wcount"] > WORD_LIM)
        """
        for t in tokens:
            if len(t) > 90:
                urlDat["page full of links"] = True
                print('premature end')
                print('premature end long word length \n\n\n\n\n\n\n\n')

                return urlDat
        """
        urlDat["tokens"] = tokens

        if len(tokens) and word_lim:  #  and server_error:
            lexicon = textstat.lexicon_count(corpus, True)
            urlDat["uniqueness"] = len(set(tokens)) / float(len(tokens))
            urlDat["unique_words"] = len(set(tokens))

            # It's harder to have a good unique ratio in a long document, as 'and', 'the' and 'a', will dominate.
            # big deltas mean redudancy/sparse information/information/density

            sentences = sent_tokenize(corpus)
            testimonial = TextBlob(corpus)
            urlDat["sp"] = testimonial.sentiment.polarity
            urlDat["ss"] = testimonial.sentiment.subjectivity
            urlDat["sp_norm"] = np.abs(testimonial.sentiment.polarity)
            urlDat["ss_norm"] = np.abs(testimonial.sentiment.subjectivity)
            urlDat["gf"] = textstat.gunning_fog(corpus)

            # explanation of metrics
            left, right = complexityAlongtheText(corpus)
            if right <= left and right !=0::
                urlDat["standard"] = right
            else:
                urlDat["standard"] = left
            if urlDat["gf"] <= urlDat["standard"] and urlDat["gf"] !=0:
                urlDat["standard"] = urlDat["gf"]
            urlDat["reading_time"] = np.round(textstat.reading_time(corpus),2)
            # urlDat["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(corpus)
            # if urlDat["flesch_kincaid_grade"] <= urlDat["standard"]:
            #    urlDat["standard"] = urlDat["flesch_kincaid_grade"]

    return urlDat


def process_dics(urlDats):
    dfs = []
    for urlDat in tqdm(urlDats):
        # pandas Data frames are best data container for maths/stats, but steep learning curve.
        # Other exclusion criteria. Exclude reading levels above grade 100,
        # as this is most likely a problem with the metric algorithm, and or rubbish data in.
        # TODO: speed everything up, by performing exclusion criteri above not here.
        if len(dfs) == 0:
            dfs = pd.DataFrame(pd.Series(urlDat)).T
        dfs = pd.concat([dfs, pd.DataFrame(pd.Series(urlDat)).T])
    return dfs
