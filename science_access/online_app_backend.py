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
from .t_analysis import text_proc
from .t_analysis import text_proc, perplexity, unigram_zipf

import streamlit as st

if 'DYNO' in os.environ:
    heroku = True
else:
    heroku = False
from time import sleep
import numpy as np

from tqdm.auto import tqdm
import streamlit as st
OPENACCESS = False


def metricss(rg):
    if isinstance(rg,list):
        pub_count = len(rg)
        mean_standard = np.mean([ r['standard'] for r in rg if 'standard' in r.keys()])
        return mean_standard
    else:
        return None
def metricsp(rg):
    if isinstance(rg,list):
        pub_count = len(rg)
        penalty = np.mean([ r['penalty'] for r in rg if 'penalty' in r.keys()])
        penalty = np.mean([ r['perplexity'] for r in rg if 'perplexity' in r.keys() ])

        return penalty
    else:
        return None

def filter_empty(the_list):
    the_list = [ tl for tl in the_list if tl is not None ]
    the_list = [ tl for tl in the_list if type(tl) is not type(str('')) ]

    return [ tl for tl in the_list if 'standard' in tl.keys() ]



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

import dask

def take_url_from_gui(author_link_scholar_link_list,tns):
    '''
    inputs a URL that's full of publication orientated links, preferably the
    authors scholar page.
    '''
    author_results = []
    #if heroku:
    #    follow_links = collect_pubs(author_link_scholar_link_list)[3:25]
    #else:
    #st.text(author_link_scholar_link_list)

    follow_links = collect_pubs(author_link_scholar_link_list)
    follow_links = follow_links[0:tns-1]
    #st.text('links empty')

    #st.text(follow_links)
    for r in tqdm(follow_links,title='Scrape in Progress. Please Wait.'):
  
        try:
            urlDat = process(r)
            author_results.append(urlDat)
        except:
            follow_more_links = collect_pubs(r)
            #lazy = (dask.delayed(process)(w) for w in follow_more_links)
            #author_results = list(dask.compute(*lazy))

            for r in tqdm(follow_more_links,title='following links from after following original links'):
                #if heroku:
                sleep(np.random.uniform(1,2))
                urlDat = process(r) 
                author_results.append(urlDat)       
    author_results = [urlDat for urlDat in author_results if not isinstance(urlDat,type(None))]
    #    author_results.append(urlDat)
    return author_results

def unigram_model(author_results):
    '''
    takes author results.
    '''
    terms = []
    for k,v in author_results.items():
        try:
            #author_results_r[k] = list(s for s in v.values()  )
            author_results[k]['files'] = list(s for s in v.values()  )

            words = [ ws['tokens'] for ws in author_results[k]['files'] if ws is not None ]
            author_results[k]['words'] = words
            terms.extend(words)# if isinstance(terms,dict) ]
        except:
            print(terms[-1])
    big_model = unigram(terms)
    with open('author_results_processed.p','wb') as file:
        pickle.dump(author_results,file)
    with open('big_model_science.p','wb') as file:
        pickle.dump(list(big_model),file)

    return big_model
'''
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
'''


def update_web_form(url,tns):
    author_results = take_url_from_gui(url,tns)
    ar =  copy.copy(author_results)
    datax = filter_empty(ar)
    met = metricss(ar)
    df = pd.DataFrame(datax)
    return df, met, author_results

def enter_name_here(scholar_page, name,tns):
    df, datay, author_results = update_web_form(scholar_page,tns)
    return df, datay, author_results

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def ar_manipulation(ar):
    ar = [ tl for tl in ar if tl is not None ]
    ar = [ tl for tl in ar if type(tl) is not type(str('')) ]
    ar = [ tl for tl in ar if 'standard' in tl.keys() ]

    with open('data/trainingDats.p','rb') as f:
        trainingDats = pickle.load(f)
        
    trainingDats.extend(ar)
    return (ar, trainingDats)
def call_from_front_end(NAME,OPENACCESS=False,tns=16):
    #if not OPENACCESS:
        #scholar_link=str('https://scholar.google.com/scholar?hl=en&as_sdt=0%2C3&q=')+str(NAME)
    #scholar_link=str('https://academic.microsoft.com/search?q=')+str(NAME)
    scholar_link=str('https://www.base-search.net/Search/Results?lookfor=')+str(NAME)

    # https://paperpile.com/g/academic-search-engines/
    _, _, ar  = enter_name_here(scholar_link,NAME,tns)
    (ar, trainingDats) = ar_manipulation(ar)
    return ar

    '''
    import os
    from crossref_commons.iteration import iterate_publications_as_json
    import requests
    from crossref.restful import Works

    if OPENACCESS:
        #queries = {'query.author': NAME}
        ar = []
        #works.query("Richard Gerkin").url
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
