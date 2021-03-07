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
import re


from textblob import TextBlob
from textstat.textstat import textstat

from science_access.utils import check_passive
from science_access.abstract_cleanup import cleanup_pretagger_all
from science_access.readabilityFunctions import countWordsSentSyl, NDC, FRE

tagger = PerceptronTagger(load=False)
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
    "author",
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


#ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words("english"))

from typing import Union
def complexityAlongtheText(str:text, int:chunk_length=128)->Union[float(),float()]:
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

def freeAlongtheText(str:text, int:chunk_length=128)->float():
    words = text.split()
    cur = 0
    stds = []
    while cur < len(words):
        sub = words[cur : cur + chunk_length]
        sub_text = " ".join(sub)
        wc, sc, sylCount, remainingText, wordLen = countWordsSentSyl(
            sub_text, ignoreSingleSentences=1
        )
        fre = FRE(wc, sc, sylCount)
        cur += chunk_length
        fres.append(fre)
    return np.mean(fres)


def get_ref(str:references):
    for nubmer, line in enumerate(references, 1):  # skip last element with page number
        line = line.strip()
        if line:  # skip empty line
            authors_and_year = re.match("((.*)\. (\d{4})\.)", line)
            if type(authors_and_year) is not type(None):
                text, authors, year = authors_and_year.groups()
                names = re.split(",[ ]*and |,[ ]*| and ", authors)
                names = [(name, name.split(" ")[-1]) for name in names]

from collections import Iterable
def text_proc(corpus, urlDat={}, WORD_LIM=60):
    if type(corpus) is type(str()) and corpus not in str(
        "Redirecting"
    ):  # and not str("privacy policy") in corpus:
        # corpus = corpus.replace("-", " ")  # remove characters that nltk can't read
        corpus = corpus.replace("/", " ")  # remove characters that nltk can't read
        corpus = corpus.lower()
        corpus = "".join([i for i in corpus if not i.isdigit()])
        corpus = re.sub(r"^https?:\/\/.*[\r\n]*", "", corpus, flags=re.MULTILINE)
        corpus = re.sub(r"^http?:\/\/.*[\r\n]*", "", corpus, flags=re.MULTILINE)
        corpus = corpus.replace("\n",""" """)  # remove characters that nltk can't read
        corpus = corpus.replace(u"\xa0", u" ")
        corpus = corpus.replace(u"\\", u" ")
        corpus = [i for i in corpus if i not in not_want_list]
        assert type(exclusive) is type(Iterable)
        #print(exclusive, "exclusive")
        # string = string.replace(u'\xa0', u' ')

        posa = corpus.lower().find("abstract")
        corpus = corpus[posa:]
        posr = corpus.lower().find("references")
        corpus = corpus[:posr]
        posb = corpus.lower().find("bibliography")
        corpus = corpus[:posb]
        corpus = cleanup_pretagger_all(corpus)
        urlDat["big_words"] = [word for word in corpus if len(word) > 40]
        ignoreSingleSentences = 1



        wc, sc, sylCount, remainingText, wordLen = countWordsSentSyl(
            corpus, ignoreSingleSentences=1
        )
        if wordLen < WORD_LIM:
            return {}
        if wordLen >= WORD_LIM:

            remainingText = " ".join(remainingText)
            remainingText = remainingText.lower()
            if wc > 0 and sc > 0:
                urlDat["standard"] = textstat.text_standard(corpus, float_output=True)
                urlDat["fre_unbiased"] = freeAlongtheText(corpus)
                if urlDat["standard"] > 50:
                    return None
                if urlDat["standard"] == 0:
                    return None

                fre = FRE(wc, sc, sylCount)
                ndc = NDC(
                    remainingText, wc, sc
                )  # calc NDC Index and Perctage Diff Words                                         #calc NDC index
                urlDat["fre"] = fre  # textstat.text_standard(corpus, float_output=True)
                urlDat["ndc"] = ndc[
                    0
                ]  # textstat.text_standard(corpus, float_output=True)
                # https://stackoverflow.com/questions/62492797/get-bibliography-list-and-its-count-from-text-python
            tokens = word_tokenize(corpus)
            wc_t, sc_t, sylCount, remainingText, wordLen = countWordsSentSyl(
                tokens, ignoreSingleSentences=ignoreSingleSentences
            )
            #print(wc_t, wc, sc_t, sc)

            urlDat["standard_unbiased"] = complexityAlongtheText(corpus, chunk_length=128)
            try:
                urlDat["concensus"] = np.mean(
                    [
                        np.mean(urlDat["fre"]),
                        np.mean(urlDat["ndc"]),
                        np.mean(urlDat["standard_unbiased"]),
                    ]
                )
            except:
                pass
            tokens = [w.lower() for w in tokens if w.isalpha()]
            tokens = [w.lower() for w in tokens]  # make everything lower case
            urlDat["wcount"] = textstat.lexicon_count(str(tokens))
            word_lim = bool(urlDat["wcount"] > WORD_LIM)
            urlDat["tokens"] = tokens

            if len(tokens):
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
