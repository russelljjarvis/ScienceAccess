from typing import List
import PyPDF2
from pathlib import Path
import copy

import semanticscholar as sch

import os.path

from collections import OrderedDict

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import streamlit as st
import numpy as np

import requests
from time import sleep

from tqdm.auto import tqdm
import streamlit as st


from .t_analysis import text_proc


class tqdm:
	"""
	this just gives streamlit tqdm's progress bar.
	"""
	def __init__(self, iterable, title=None):
		if title:
			st.write(title)

		self.prog_bar = st.progress(0)

		self.iterable = iterable
		self.length = len(iterable)
		self.i = 0

	def __iter__(self):
		for obj in self.iterable:
			yield obj
			self.i += 1
			current_prog = self.i / self.length
			self.prog_bar.progress(current_prog)


def author_to_urls(NAME):
	"""
	splash_url is a URL where Dissemin thinks that the paper is described, without being necessarily available. This can be a publisher webpage (with the article available behind a paywall), a page about the paper without a copy of the full text (e.g., a HAL page like https://hal.archives-ouvertes.fr/hal-01664049), or a page from which the paper was discovered (e.g., the profile of a user on ORCID).
	pdf_url is a URL where Dissemin thinks the full text can be accessed for free. This is rarely a direct link to an actual PDF file, i.e., it is often a link to a landing page (e.g., https://arxiv.org/abs/1708.00363). It is set to null if we could not find a free source for this paper.
	"""
	response = requests.get("https://dissem.in/api/search/?authors=" + str(NAME))
	author_papers = response.json()
	visit_urls = []
	coauthors = []
	titles = []
	for p in author_papers["papers"]:
		coauthors_ = p["authors"]
		title = p["title"]
		titles.append(title)
		coauthors.append(coauthors_)
		records = p["records"][0]
		if "pdf_url" in records.keys():
			visit_urls.append(records["pdf_url"])

		if "splash_url" in records.keys():
			visit_urls.append(records["splash_url"])
		if "doi" in records.keys():
			visit_urls.append(records["doi"])

	visit_urls = [i for i in visit_urls if "FIGSHARE" not in i]
	visit_urls = [i for i in visit_urls if "figshare" not in i]
	dois = []

	for link in visit_urls:
		if "https://doi.org/" in link:
			li = link.split("https://doi.org/")
			dois.append(li[1])
		if "http://dx.doi.org" in link:
			li = link.split("http://dx.doi.org")
			dois.append(li[1])
	return dois, coauthors, titles, visit_urls


def check_link(link):
	if link is not None:
		urlDatTemp = process(link)
		return urlDatTemp


import dask
from dask import delayed, compute


def visit_link(NAME, tns):  # , more_links):
	"""
	inputs a URL that's full of publication orientated links, preferably the
	authors scholar page.
	"""

	author_results = []

	author_results, visit_more_urls = visit_link_unpaywall(NAME)

	# visit_urls.extend(more_links)
	# for index, link in enumerate(
	#    tqdm(visit_urls, title="Text mining via API calls. Please wait.")
	# ):
	# for i in visit_urls[0:tns]:
	#    author_results.append(dask.delayed(visit_link_unpaywall)(i))#.compute()

	#        author_results.append(urlDatTemp)
	author_results = [
		urlDat for urlDat in list(author_results) if not isinstance(urlDat, type(None))
	]
	# for urlDat in author_results:
	#    st.markdown(urlDat)

	return author_results, visit_more_urls


def semantic_scholar_alias(NAME):
	"""
	inputs a URL that's full of publication orientated links, preferably the
	authors scholar page.
	"""

	author_results = []
	aliases = None
	dois, coauthors, titles, visit_urls = author_to_urls(NAME)
	inv_alias_dict = {}
	velocity = {}
	for d in dois:
		paper = sch.paper(d, timeout=32)
		if "authors" in paper.keys():
			all_coauthors = paper["authors"]
			for co_name in all_coauthors:
				key = co_name["name"]
				if (
					NAME.split(" ")[0] in key.split(" ")[0]
					or key.split(" ")[0] in NAME.split(" ")[0]
					or NAME.split(" ")[-1] in key.split(" ")[-1]
				):
					author = sch.author(co_name["authorId"], timeout=32)

					if "aliases" in author.keys():
						aliases = author["aliases"]
						return aliases


import streamlit as st


def visit_semantic_scholar_abstracts(NAME, tns, more_links):
    """
    inputs a URL that's full of publication orientated links, preferably the
    authors scholar page.
    """

    author_results = []
    aliases = None
    dois, coauthors, titles, visit_urls = author_to_urls(NAME)

    for d in tqdm(dois, title="visiting abstracts"):
        paper = sch.paper(d, timeout=8)

        urlDat = {}
        if "citationVelocity" in paper.keys():
            urlDat["citationVelocity"] = paper["citationVelocity"]
        if "fieldsOfStudy" in paper.keys():
            urlDat["fieldsOfStudy"] = paper["fieldsOfStudy"]
        if "numCitedBy" in paper.keys():
            urlDat["numCitedBy"] = paper["numCitedBy"]
        #urlDat["influentialCitationCount"] = paper["influentialCitationCount"]
        urlDat["semantic"] = True

        if "url" in paper.keys():
            urlDat["link"] = paper["title"]
        if aliases is None:
            if "aliases" in paper.keys():
                urlDat["aliases"] = paper["aliases"]
            else:
                pass
        if "abstract" in paper.keys():
            urlDat = text_proc(str(paper["abstract"]), urlDat)
            author_results.append(urlDat)
    author_results = [
        urlDat for urlDat in author_results if not isinstance(urlDat, type(None))
    ]

    return author_results, visit_urls


from dask import compute


def visit_link_unpaywall(NAME):  # ), tns, visit_urls):
	"""
	inputs a URL that's full of publication orientated links, preferably the
	authors scholar page.
	"""
	author_results = []
	dois, coauthors, titles, visit_urls = author_to_urls(NAME)
	if len(visit_urls) > 30:
		visit_urls = visit_urls[0:29]
		st.warning("too many publications >40 truncating list")
	for index, link in enumerate(
		tqdm(visit_urls, title="Buildiing suitable links.")
	):
		urlDat = dask.delayed(process)(link)
		author_results.append(urlDat)
	author_results = list(dask.compute(author_results))
	if len(filter_empty(author_results)):
		return author_results, visit_urls
	else:
		for index, link in enumerate(
			tqdm(visit_urls, title="Visiting links to extract text from full text.")
		):

			urlDat = process(link)
			author_results.append(urlDat)
	return author_results, visit_urls


def unpaywall_semantic_links(NAME, tns, fast=True):
	"""
	inputs a URL that's full of publication orientated links, preferably the
	authors scholar page.
	"""
	dois, coauthors, titles, visit_urls = author_to_urls(NAME)
	visit_more_urls = []
	for index, doi_ in enumerate(dois):, title="Building suitable links.")):
		if fast:
			r0 = str("https://api.semanticscholar.org/") + str(doi_)
			visit_more_urls.append(r0)
		if not fast:

			r = (
				str("https://api.unpaywall.org/v2/")
				+ str(doi_)
				+ str("?email=russelljarvis@protonmail")
			)
			response = requests.get(r)
			response = response.json()
			if "oa_locations" in response.keys():
				res_list = response["oa_locations"]
				for res in res_list:
					if "url_for_pdf" in res.keys():
						res_ = res["url_for_pdf"]
						visit_more_urls.append(res_)

			if "url_for_landing_page" in response.keys():
				res = response["url_for_landing_page"]
				visit_more_urls.append(res)

			if "doi_url" in response.keys():
				res = response["doi_url"]
				visit_more_urls.append(res)
	return visit_more_urls


def convert_pdf_to_txt(content, verbose=False):
	# https://github.com/allenai/science-parse/blob/master/server/README.md
	# os.subprocess(curl -v -H "Content-type: application/pdf" --data-binary @paper.pdf "http://scienceparse.allenai.org/v1")
	try:
		pdf = io.BytesIO(content.content)
	except:
		pdf = io.BytesIO(content)
	parser = PDFParser(pdf)
	if (type(parser) is not type(bytes)) and (type(parser) is not type(None)):
		document = PDFDocument(parser, password=None)  # this fails
		write_text = ""
		for page in PDFPage.create_pages(document):
			interpreter.process_page(page)
			write_text += " " + retstr.getvalue() + " "
		# Process all pages in the document
		text = str(write_text)
		mean_word_len = np.mean([len(t) for t in text])
		if mean_word_len > 33:
			return str("")

		if verbose:
			st.text(text)
		return text
	else:
		return str("")


import html


def process(link):
	urlDat = {}
	if link is None:
		return None
	try:
		response = requests.get(link)
	except:
		urlDat["link"] = link
		return urlDat
	if str("pdf") not in link:
		buffered = ""
		response = requests.get(link)

		#crude_html = html.unescape(response.text)
		crude_html = response.text
		soup = BeautifulSoup(crude_html, "html.parser")
		for script in soup(["script", "style"]):
			script.extract()  # rip it out

		text = soup.get_text()
		lines = (
			line.strip() for line in text.splitlines()
		)  # break into lines and remove leading and trailing space on each
		chunks = (
			phrase.strip() for line in lines for phrase in line.split("  ")
		)  # break multi-headlines into a line each
		text = "\n".join(chunk for chunk in chunks if chunk)  # drop blank lines
		buffered = str(text)

	else:

		try:
			fname = str(link[0:9])+str(".pdf")

			filename = Path(fname)
			response = requests.get(link, timeout=10)

			filename.write_bytes(response.content)

			reader = PyPDF2.PdfFileReader(fname)
			buffered = ""
			for p in range(1, reader.numPages):
				buffered += str(reader.getPage(p).extractText())

		except:
			buffered = ""

	urlDat["link"] = link
	urlDat = text_proc(buffered, urlDat)
	return urlDat


def update_web_form(NAME, tns, fast=True):
	if fast:
		more_links = unpaywall_semantic_links(NAME, tns, fast=True)

		author_results, visit_urls_temp = visit_semantic_scholar_abstracts(
			NAME, tns, more_links
		)
		author_results = copy.copy(author_results)
		datax = filter_empty(author_results)
		df = pd.DataFrame(datax)
		met = metricss(author_results)
	else:
		author_results, visited_urls = visit_link(NAME, tns)
		df = pd.DataFrame(author_results)
		met = metricss(author_results)
	return df, met, author_results



def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx


import pickle


def ar_manipulation(ar: List = []):
	ar = [tl for tl in ar if tl is not None]
	ar = [tl for tl in ar if type(tl) is type(dict())]
	ar = [tl for tl in ar if type(tl) is not type(str(""))]
	ar = [tl for tl in ar if "standard" in tl.keys()]
	with open("trainingDats.p", "rb") as f:
		trainingDats = pickle.load(f)

	trainingDats.extend(ar)
	return (ar, trainingDats)




def call_from_front_end(NAME="", OPENACCESS=True, tns=16, fast=True):
	df, datay, ar = update_web_form(NAME, tns, fast=fast)
	(ar, trainingDats) = ar_manipulation(ar)
	return ar


def metricss(rg):
	if isinstance(rg, list):
		pub_count = len(rg)
		rg = [r for r in rg if type(r) is type(dict())]
		mean_standard = np.mean([r["standard"] for r in rg if "standard" in r.keys()])
		return mean_standard
	else:
		return None


def metricsp(rg):
	if isinstance(rg, list):
		pub_count = len(rg)
		penalty = np.mean([r["penalty"] for r in rg if "penalty" in r.keys()])
		penalty = np.mean([r["perplexity"] for r in rg if "perplexity" in r.keys()])

		return penalty
	else:
		return None


def filter_empty(the_list):
	the_list = [tl for tl in the_list if tl is not None]
	the_list = [tl for tl in the_list if type(tl) is not type(str(""))]
	the_list = [tl for tl in the_list if type(tl) is type(dict())]
	return [tl for tl in the_list if "standard" in tl.keys()]
