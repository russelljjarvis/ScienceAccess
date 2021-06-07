from typing import List
import PyPDF2
from pathlib import Path
import copy

import semanticscholar as sch

import os.path

# import pdb
# import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import streamlit as st

from time import sleep
import numpy as np

from tqdm.auto import tqdm
import streamlit as st

# import dask
import requests

from .t_analysis import text_proc


class tqdm:
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
    # visit_urls = [i for i in visit_urls if "doi" in i]
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
    # dois, coauthors, titles, visit_urls
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
    # dois, coauthors, titles
    dois, coauthors, titles, visit_urls = author_to_urls(NAME)
    # alias_dict = {}
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
    # for index, doi_ in enumerate(tqdm(dois, title="Building Suitable Links")):

    for d in tqdm(dois, title="Visiting links to extract text from the abstract."):
        paper = sch.paper(d, timeout=8)
        urlDat = {}
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
    #if len(visit_urls) > 50:
        #visit_urls = visit_urls[0:49]
        #st.warning("too many publications >40 truncating list")
    for index, link in enumerate(
        tqdm(visit_urls, title="Building suitable links.")
    ):
        urlDat = dask.delayed(process)(link)
        author_results.append(urlDat)
    author_results = list(dask.compute(author_results))
    if len(filter_empty(author_results)):
        return author_results, visit_urls
    else:
        for index, link in enumerate(
            tqdm(visit_urls, title="Visiting links to extract full text.")
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
    for index, doi_ in enumerate(tqdm(dois, title="Building suitable links.")):
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


def process(link):  # , REDIRECT=False):
    urlDat = {}

    if link is None:
        return None
    try:
        response = requests.get(link)
    except:
        urlDat["link"] = link

        return urlDat
    if str("pdf") not in link:
        # try:
        buffered = ""
        response = requests.get(link)
        # crude_html = response.json()
        # st.success("html hanged...")

        # crude_html = html.unescape(response.text)
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
        # except:

    else:

        try:

            filename = Path("this_pdf.pdf")
            response = requests.get(link, timeout=10)

            filename.write_bytes(response.content)

            reader = PyPDF2.PdfFileReader("this_pdf.pdf")
            buffered = ""
            for p in range(1, reader.numPages):
                buffered += str(reader.getPage(p).extractText())

        except:
            buffered = ""
        # st.success("pdf worked")

    urlDat["link"] = link
    urlDat = text_proc(buffered, urlDat)
    # st.success("some entity processed")

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
        # st.write(df)

        met = metricss(author_results)
        # st.markdown(met)
    return df, met, author_results


""""
def update_web_form_full_text(NAME, tns):
    #more_links = unpaywall_semantic_links(NAME, tns)
    #author_results_temp, visit_urls_temp = visit_semantic_scholar_abstracts(
    #    NAME, tns, more_links
    #)
    #urls_to_visit = unpaywall_semantic_links(NAME, tns,fast=False)#, more_links)
    author_results, visited_urls = visit_link(NAME, tns)
    #author_results.extend(author_results_temp)
    #ar = copy.copy(author_results)
    #datax = filter_empty(ar)
    #met = metricss(ar)
    #df = pd.DataFrame(datax)
    return author_results, visited_urls

"""
"""
def enter_name_here(scholar_page, name, tns,fast=True):
    df0, datay, author_results0 = update_web_form_full_text(scholar_page, tns,fast=fast)
    #except:
    #df1, datay, author_results1 = update_web_form(scholar_page, tns)
    #df = pd.concat([df0,df1])
    #author_results=author_results0
    #author_results.extend(author_results1)
    return df0, datay, author_results0
"""


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


# def call_from_front_end(NAME: str = "", OPENACCESS: bool = True, tns: int = 16):


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


# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 14:58:01 2019
@author: Ken

import arxiv
import pandas as pd
import requests

result = arxiv.query(query="all:deep learning")
data = pd.DataFrame(columns = ["title","id",'arxiv_url','published'])

print("got arxiv data")

for i in range(len(result)):
  id = result[i]['id'].split("/")[-1].split("v")[0]
  title = result[i]['title']
  arxiv_url = result[i]['arxiv_url']
  published = result[i]['published']
  data_tmp = pd.DataFrame({"title":title,"id":id, "arxiv_url":arxiv_url, "published":published},index=[0])
  data = pd.concat([data,data_tmp]).reset_index(drop=True)

  print("get citations")
  citation_num_list = []
  for num, i in enumerate(data["id"]):
	 if num % 10 == 0:
	   print(num)
  try:
	sem = requests.get("https://api.semanticscholar.org/v1/paper/arXiv:"+i).json()
	citation_num = len(sem["citations"])
  except:
	citation_num = 0
  citation_num_list.append(citation_num)

data["citation"] = citation_num_list

data = data.sort_values(by='citation', ascending=False)

data.to_csv("data.csv",index=False)


	ar = []
	works = Works()
	ww =  works.query(author=NAME).filter(from_online_pub_date='2000')
		# urls
		#bi =[p for p in iterate_publications_as_json(max_results=100, queries=queries)]
		#for p in bi[0:30]:
		cnt = 0
		#from tqdm.auto import tqdm
		#if 'Abstract' in corpus:
		first = NAME.split(" ",1)[0]
		last = NAME.split(" ",1)[1]
		from pybliometrics.scopus import ScopusSearch
		s = ScopusSearch('AUTHLASTNAME( {1} )'.format(NAME))
		import pandas as pd
		df = pd.DataFrame(pd.DataFrame(s.results))
		st.write(df)
		#from pybliometrics.scopus import AuthorSearch
		#s = AuthorSearch('AUTHLAST({0}) and AUTHFIRST({1})'.format(first,last))


		#pbar = tqdm(total=tns,title='Scrape in Progress. Please Wait.')
		prog_bar = st.progress(0)

		for p in df['doi']:#tqdm(ww.all(),title='Scrape in Progress. Please Wait.'):
			if cnt>=tns:
				break
			if p['DOI']:
				res = str('https://api.unpaywall.org/v2/')+str(p['DOI'])+str('?email=rjjarvis@asu.edu')

						  #“https://api.unpaywall.org/v2/" +value +”?email=your@emaildomain.com”
				response = requests.get(res)
				response = response.json()
				st.text(response['doi_url'])
				st.text(response['data_standard'])
				if response['is_oa'] and response is not None:
					st.text(response.keys())
					#url = response['free_fulltext_url']
					url = response['best_oa_location']['url']#['url_for_pdf']
					st.text(url)
					if url is not None:
						urlDat = process(url)
						if not isinstance(type(urlDat),type(None)):
							if NAME in urlDat['tokens']:
								cnt+=1
								current_prog = cnt/ tns
								prog_bar.progress(current_prog)
								ar.append(urlDat)


			#Abstract only
			elif 'URL' in p.keys():
				temp = p['URL']
				urlDat = process(temp)
				#st.text(urlDat['tokens'])
				if not isinstance(type(urlDat),type(None)):
					if 'tokens' in urlDat.keys():
						if NAME in urlDat['tokens']:
							ar.append(urlDat)
							cnt+=1
							current_prog = cnt/ tns
							prog_bar.progress(current_prog)


		(ar, trainingDats) = ar_manipulation(ar)
		#pbar.close()

	with open('data/traingDats.p','rb') as f:
		trainingDats_old = pickle.load(f)
	trainingDats.extend(trainingDats_old)
	with open('data/traingDats.p','wb') as f:
		pickle.dump(trainingDats,f)
	'''
"""

"""

def brian_function(author_link_scholar_link_list, tns):
    inputs a URL that's full of publication orientated links, preferably the
    authors scholar page.
    from bs4 import BeautifulSoup

    author_results = []
    follow_links = collect_pubs(author_link_scholar_link_list)
    follow_links = follow_links[0 : tns - 1]
    for r in tqdm(follow_links, title="Scrape in Progress. Please Wait."):
        urlDat = process(r)
        # soup = BeautifulSoup(document, 'html.parser')

        # try:
        #        urlDat = process(r)
        #    author_results.append(urlDat)
        # except:
        #    follow_more_links = collect_pubs(r)
        #    for r in tqdm(follow_more_links,title='following links from after following original links'):
        #        sleep(np.random.uniform(1,2))
        #        urlDat = process(r)
        #        author_results.append(urlDat)
        # author_results = [urlDat for urlDat in author_results if not isinstance(urlDat,type(None))]
    return author_results


def unigram_model(author_results):
    #takes author results.
    #
    terms = []
    for k, v in author_results.items():
        try:
            # author_results_r[k] = list(s for s in v.values()  )
            author_results[k]["files"] = list(s for s in v.values())

            words = [
                ws["tokens"] for ws in author_results[k]["files"] if ws is not None
            ]
            author_results[k]["words"] = words
            terms.extend(words)  # if isinstance(terms,dict) ]
        except:
            print(terms[-1])
    big_model = unigram(terms)
    with open("author_results_processed.p", "wb") as file:
        pickle.dump(author_results, file)
    with open("big_model_science.p", "wb") as file:
        pickle.dump(list(big_model), file)

    return big_model


Not used
def info_models(author_results):
	big_model = unigram_model(author_results)
	compete_results = {}
	for k,v in author_results.items():
		per_dpc = []
		try:
			for doc in author_results[k]['words']:
				per_doc.append(perplexity(doc, big_model))
		except:
			pass
		compete_results[k] = np.mean(per_doc)
		author_results[k]['perplexity'] = compete_results[k]
	return author_results, compete_results
"""
