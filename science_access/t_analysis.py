# Scientific readability project
# authors ...,
# Russell Jarvis
# https://github.com/russelljjarvis/
# russelljarvis@protonmail.com

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
#from nltk.tokenize import word_tokenize

from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd
import re
import streamlit as st
from .utils import black_string, clue_links, clue_words, comp_ratio, publication_check
import re

from collections import Iterable

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
def complexityAlongtheText(text:str, chunk_length:int=5)->Union[float,float,str]:
	words = sent_tokenize(text)
	#words = #text.split()
	cur = 0
	stds = []
	hardest_chunk_index = 0
	while cur < len(words):
		sub = words[cur : cur + 5]
		sub_text = " ".join(sub)
		std = textstat.text_standard(sub_text, float_output=True)
		cur += chunk_length
		if std>hardest_chunk_index:
			hardest_chunk_index = cur
		stds.append(std)
	hard_snippet = words[hardest_chunk_index : hardest_chunk_index + chunk_length]
	hs=""
	for h in hard_snippet:
		hs+=h+str(" ")
	#st.text(hs)
	return np.mean(stds), textstat.text_standard(text, float_output=True), hs

def freeAlongtheText(text:str, chunk_length:int=5)->float:
	#words = text.split()
	words = sent_tokenize(text)

	cur = 0
	stds = []
	fres = []
	while cur < len(words):
		sub = words[cur : cur + chunk_length]
		sub_text = " ".join(sub)
		wc, sc, sylCount, remainingText, wordLen = countWordsSentSyl(
			sub_text, ignoreSingleSentences=1
		)
		try:
			fre = FRE(wc, sc, sylCount)
			fres.append(fre)
		except:
			pass
		cur += chunk_length
	return np.mean(fres)


def get_ref(references:str):
	for nubmer, line in enumerate(references, 1):  # skip last element with page number
		line = line.strip()
		if line:  # skip empty line
			authors_and_year = re.match("((.*)\. (\d{4})\.)", line)
			if type(authors_and_year) is not type(None):
				text, authors, year = authors_and_year.groups()
				names = re.split(",[ ]*and |,[ ]*| and ", authors)
				names = [(name, name.split(" ")[-1]) for name in names]

def text_proc(corpus, urlDat={}, WORD_LIM=60,verbose=False):
	if type(corpus) is type(str()) and corpus not in str(
		"Redirecting"
	):  # and not str("privacy policy") in corpus:


		if str("some error has occurred while processing your request") in corpus:
			return {}
		if str("We apologize for the inconvenience...") in corpus:
			return {}
		if np.mean([ len(w) for w in corpus ])>35:
			return {}

		corpus = corpus.replace("/", " ")  # remove characters that nltk can't read
		corpus = corpus.lower()
		corpus = corpus.replace(u"\xa0", u" ")
		corpus = corpus.replace(u"\\", u" ")
		if not "significance" in corpus and not "abstract" in corpus:
			return {}
		if "abstract" in corpus:
			posa = corpus.lower().find("abstract")
			corpus = corpus[posa:]
			posr = corpus.lower().find("references")
			corpus = corpus[:posr]
			posb = corpus.lower().find("bibliography")
			corpus = corpus[:posb]
		if "significance" in corpus:
			poss = corpus.lower().find("significance")
			corpus = corpus[poss:]
			posp = corpus.lower().find("purpose")
			corpus = corpus[:posp]

		urlDat["big_words"] = [word for word in corpus if len(word) > 40]
		ignoreSingleSentences = 1

		corpus = cleanup_pretagger_all(corpus)
		if verbose:
			st.text('pretagger all')
			st.text(type(corpus))

		tokens = word_tokenize(corpus)
		if verbose:
			st.text("token input")
			st.text(tokens)
		tokens = [ t for t in tokens if t not in not_want_list]
		if np.mean([ len(t) for t in tokens ])>40:
			return {}
		tokens = [ t for t in tokens if len(t)<40 ]
		if verbose:
			st.text("token input")
			st.text(tokens)
		wc, sc, sylCount, remainingText, wordLen = countWordsSentSyl(
			tokens, ignoreSingleSentences=1
		)

		if len(tokens) < WORD_LIM:
			return {}
		if len(tokens) >= WORD_LIM:

			remainingText = " ".join(remainingText)
			remainingText = remainingText.lower()
			if wc > 0 and sc > 0:
				#urlDat["standard"] = textstat.text_standard(corpus, float_output=True)
				meanv,total,hard_snippet = complexityAlongtheText(corpus, chunk_length=128)
				urlDat["standard_unbiased"] = meanv
				urlDat["standard"] = total

				urlDat["hard_snippet"] = hard_snippet
				urlDat["fre_unbiased"] = freeAlongtheText(corpus)

				#    return None

				fre = FRE(wc, sc, sylCount)
				ndc = NDC(
					remainingText, wc, sc
				)  # calc NDC Index and Perctage Diff Words                                         #calc NDC index
				urlDat["fre"] = fre  # textstat.text_standard(corpus, float_output=True)
				urlDat["ndc"] = ndc[
					0
				]  # textstat.text_standard(corpus, float_output=True)
				# https://stackoverflow.com/questions/62492797/get-bibliography-list-and-its-count-from-text-python

			wc_t, sc_t, sylCount, remainingText, wordLen = countWordsSentSyl(
				tokens, ignoreSingleSentences=ignoreSingleSentences
			)
			if urlDat["fre_unbiased"]< urlDat["standard"] and urlDat["fre_unbiased"]>0:
				urlDat["standard"] = urlDat["fre_unbiased"]
			if urlDat["standard_unbiased"]< urlDat["standard"]  and urlDat["fre_unbiased"]>0:
				urlDat["standard"] = urlDat["standard_unbiased"]
			if urlDat["standard"] > 65:
				return None

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

				testimonial = TextBlob(corpus)
				urlDat["sp"] = testimonial.sentiment.polarity
				urlDat["ss"] = testimonial.sentiment.subjectivity
				urlDat["sp_norm"] = np.abs(testimonial.sentiment.polarity)
				urlDat["ss_norm"] = np.abs(testimonial.sentiment.subjectivity)
				urlDat["gf"] = textstat.gunning_fog(corpus)
	if "standard" in urlDat.keys():
		if urlDat["standard"] == 0:
			return None

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
