import copy
import matplotlib.pyplot as plt
import seaborn as sns
import os.path
import pdb
import pickle
from collections import OrderedDict

import IPython.display as d
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from .crawl import collect_pubs
from .get_bmark_corpus import process

# from .t_analysis import text_proc
# from .t_analysis import text_proc, perplexity, unigram_zipf

import streamlit as st

if "DYNO" in os.environ:
    heroku = True
else:
    heroku = False
from time import sleep
import numpy as np

from tqdm.auto import tqdm
import streamlit as st
import dask
import requests


def metricss(rg):
    if isinstance(rg, list):
        pub_count = len(rg)
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

    return [tl for tl in the_list if "standard" in tl.keys()]


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


import crossref_commons.retrieval


def author_to_affiliations(NAME):
    response = requests.get("https://dissem.in/api/search/?authors=" + str(NAME))
    author_papers = response.json()
    visit_urls = []
    coauthors = []
    titles = []
    affilations = {}
    for p in author_papers["papers"]:
        coauthors_ = p["authors"]
        records = p["records"][0]
        if "doi" in records.keys():
            visit_urls.append(records["doi"])
            try:
                doi_to_affil = crossref_commons.retrieval.get_publication_as_json(
                    records["doi"]
                )
                key = stored["author"][0]["given"] + stored["author"][0]["family"]
                affilations[key] = doi_to_affil["author"][0]["affiliation"]
            except:
                pass
    return affilations


def author_to_urls(NAME):
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
        if "pdf_url" in p.keys():
            visit_urls.append(p["pdf_url"])
        records = p["records"][0]
        if "splash_url" in records.keys():
            visit_urls.append(records["splash_url"])
        if "doi" in records.keys():
            visit_urls.append(records["doi"])

    visit_urls = [i for i in visit_urls if "FIGSHARE" not in i]
    visit_urls = [i for i in visit_urls if "figshare" not in i]
    visit_urls = [i for i in visit_urls if "doi" in i]
    dois = []

    for link in visit_urls:
        if "https://doi.org/" in link:
            li = link.split("https://doi.org/")
            dois.append(li[1])
        if "http://dx.doi.org" in link:
            li = link.split("http://dx.doi.org")
            dois.append(li[1])
    return dois, coauthors, titles, visit_urls


# def elsevier():
# MY_API_KEY = str("6fae5b65e32e1c403792a2c7301ebfe7")
# r = str("https://api.elsevier.com/content/article/doi/")+str(doi_)+str("?view=FULL")+str("?APIKey=")+MY_API_KEY
# response = requests.get(r)
# print(response.status_code)
# if response.status_code>1 and response.status_code<3:
# 	print(response.content)
# 	urlDat = process(response.content)
# 	print('elsevier worked')
# else:


def take_url_from_gui(NAME, tns):
    """
    inputs a URL that's full of publication orientated links, preferably the
    authors scholar page.
    """
    author_results = []
    dois, coauthors, titles, visit_urls = author_to_urls(NAME)
    for index, doi_ in enumerate(
        tqdm(visit_urls, title="Text mining via API calls. Please wait.")
    ):
        link = doi_  # visit_urls[index]
        print(doi_, "visited \n\n\n\n\n")
        urlDatTemp = process(doi_)
        author_results.append(urlDatTemp)
    author_results = [
        urlDat for urlDat in author_results if not isinstance(urlDat, type(None))
    ]
    return author_results, visit_urls

    """
	if "tokens" in urlDatTemp.keys():
		print(urlDatTemp["tokens"])
	r = (
		str("https://api.unpaywall.org/v2/")
		+ str(doi_)
		+ str("?email=russelljjarvis@protonmail.com")
	)
	response = requests.get(r)
	response = response.json()
	urlDat = None
	if "url_for_pdf" in response.keys():
		res = response["url_for_pdf"]
		urlDat = process(res)
	if "url_for_landing_page" in response.keys() and urlDat is None:
		res = response["url_for_landing_page"]
		urlDat = process(res)
	if "doi_url" in response.keys() and urlDat is None:
		res = response["doi_url"]
		urlDat = process(res)

	if urlDat is None:
		if "tokens" in urlDatTemp.keys():
			# if len(urlDat["tokens"])<len(urlDatTemp["tokens"]):
			urlDat = urlDatTemp
	"""


def take_url_from_gui_unpaywall(NAME, tns, visit_urls):
    """
    inputs a URL that's full of publication orientated links, preferably the
    authors scholar page.
    """
    author_results = []
    dois, coauthors, titles, visit_urls = author_to_urls(NAME)
    # unpaywall_links =
    visit_more_urls = []
    for index, doi_ in enumerate(
        tqdm(dois, title="Text mining via API calls. Please wait.")
    ):
        # link = visit_urls[index]
        # urlDatTemp = process(link)
        # if "tokens" in urlDatTemp.keys():
        #    print(urlDatTemp["tokens"])
        r = (
            str("https://api.unpaywall.org/v2/")
            + str(doi_)
            + str("?email=rjjarvis@asu.edu")
        )
        response = requests.get(r)
        response = response.json()
        urlDat = None
        if "url_for_pdf" in response.keys():
            res = response["url_for_pdf"]
            # if res not in set(visit_urls):
            visit_more_urls.append(res)

            urlDat = process(res)
        if "url_for_landing_page" in response.keys() and urlDat is None:
            res = response["url_for_landing_page"]
            visit_more_urls.append(res)

            urlDat = process(res)
        if "doi_url" in response.keys() and urlDat is None:
            res = response["doi_url"]
            visit_more_urls.append(res)

            urlDat = process(res)

        if urlDat is None:
            if "tokens" in urlDatTemp.keys():
                # if len(urlDat["tokens"])<len(urlDatTemp["tokens"]):
                urlDat = urlDatTemp
        author_results.append(urlDat)
    author_results = [
        urlDat for urlDat in author_results if not isinstance(urlDat, type(None))
    ]
    return author_results, visit_more_urls


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


def update_web_form(NAME, tns):
    # author_results = brian_function(url,tns)

    author_results, visit_urls = take_url_from_gui(NAME, tns)
    author_results, visit_more_urls = take_url_from_gui_unpaywall(NAME, tns, visit_urls)
    print(set(visit_urls) & set(visit_more_urls))
    ar = copy.copy(author_results)
    datax = filter_empty(ar)
    met = metricss(ar)
    df = pd.DataFrame(datax)
    return df, met, author_results


def enter_name_here(scholar_page, name, tns):
    df, datay, author_results = update_web_form(scholar_page, tns)
    return df, datay, author_results


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def ar_manipulation(ar):
    ar = [tl for tl in ar if tl is not None]
    ar = [tl for tl in ar if type(tl) is not type(str(""))]
    ar = [tl for tl in ar if "standard" in tl.keys()]

    with open("data/trainingDats.p", "rb") as f:
        trainingDats = pickle.load(f)

    trainingDats.extend(ar)
    return (ar, trainingDats)


def call_from_front_end(NAME, OPENACCESS=True, tns=16):
    # if not OPENACCESS:
    # scholar_link=str('https://scholar.google.com/scholar?hl=en&as_sdt=0%2C3&q=')+str(NAME)
    # scholar_link=str('https://academic.microsoft.com/search?q=')+str(NAME)
    # scholar_link=str('https://www.base-search.net/Search/Results?lookfor=')+str(NAME)

    # https://paperpile.com/g/academic-search-engines/
    df, datay, ar = update_web_form(NAME, tns)

    # _, _, ar  = enter_name_here(scholar_link,NAME,tns)
    (ar, trainingDats) = ar_manipulation(ar)
    return ar


def call_from_front_end_oa(NAME, OPENACCESS=False, tns=16):
    import os
    from crossref_commons.iteration import iterate_publications_as_json
    import requests
    from crossref.restful import Works

    url = "https://pub.orcid.org/v3.0/csv-search/?q=" + str(NAME)
    # https://pub.orcid.org/v3.0/csv-search/?q=
    response = requests.get(url)

    import requests
    import pandas as pd
    import io

    urlData = requests.get(url).content
    #    rawData = pd.read_csv(io.StringIO(urlData.decode('utf-8')))

    # global url;
    """
	url = "firstURL"
	At the end of your first function you can change the value of the variable to the new URL:

	url = driver.current_url
	And then you can get the new url at the beginning of your second function:
	"""
    driver.get(urlData)


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
