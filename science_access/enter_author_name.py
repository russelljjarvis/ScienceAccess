#SComplexity.t_analysis
#from SComplexity import online_app_backend
#import argparse

import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import pickle
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import copy
#import nltk

import streamlit as st
import math
import scipy
from types import MethodType
from nltk import word_tokenize


from science_access.t_analysis import not_want_list
from science_access.online_app_backend import call_from_front_end
from science_access.online_app_backend import ar_manipulation
from science_access.word_cloud_by_word_len import generate_from_lengths
from science_access.utils import check_passive
import plotly.graph_objects as go 


def frame_to_lists(ar):
    scraped_labels = [ str(x['link']) for x in ar]
    standard_sci = [ t['standard'] for t in ar ]
    return scraped_labels, standard_sci

def try_and_update_cache(ar,trainingDats):
    '''
    Try to make the distribution accumulate information based on future queries.
    '''
    with open('data/trainingDats.p','wb') as f:
        st.write(str(type(trainingDats)))
        st.write(str(type(ar)))
        st.write('if types are data frame/list wrangling will be required')
        trainingDats.extend(ar)
        pickle.dump(f,trainingDats)
def get_table_download_link(df):
    """Generates a link allowing the data 
    in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'


st.cache 
def art_cloud(acorpus):

    # Generate a word cloud image
    WC = WordCloud(background_color="white")
    
    fig = plt.figure()
    # increase resolution by changing figure size
    #figsize=(25,25))
    wordcloud = WC.generate(acorpus)
    # interpolation "nearest decreases resolution."
    plt.imshow(wordcloud,aspect="auto", interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    return wordcloud,fig,plt

def fast_art_cloud(acorpus):
    # uses cache
    wordcloud,fig,plt = art_cloud(acorpus)
    st.pyplot()
    #    st.pyplot(width =517)

def create_giant_strings(ar,not_want_list):
    sci_corpus = ''
    for t in ar:
        if 'tokens' in t.keys():
            for s in t['tokens']:
                if s not in not_want_list:
                    if "." in s:
                        temp = s.split(".")#, " ")
                        sci_corpus+=str(' ')+temp[0]
                        sci_corpus+=str(' ')+temp[1]
                    if s not in set(not_want_list):
                        sci_corpus+=str(' ')+s#+str(' ')
    return sci_corpus
def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    text = link#.split('=')[1]
    return f'<a target="_blank" href="{link}">{text}</a>'

def extra_options(ar,trainingDats,df1):

    try:
        # try and update underlying distribution with query, so information about science 
        # is culmulative, dynamic.
        try_and_update_cache(ar,trainingDats)
    except:
        st.markdown(str(type(trainingDats)))
        st.markdown(str(type(ar)))
        st.markdown('if types are data frame/list wrangling will be required')
        st.markdown('update cache fails, data wrangling required')
    try:
        # Try to allow researchers of the app to download the data.
        # Via GUI prompts.
        st.markdown(get_table_download_link(df1), unsafe_allow_html=True)
    except:
        st.markdown('try and allow user to download data')

def grab_data_for_splash(trainingDats):

    bio_chem = [ t['standard'] for t in trainingDats ]
    biochem_labels =  [ x['file_name'] for x in trainingDats if 'file_name' in x.keys()]
    biochem_labels = [x.split("/")[-1] for x in biochem_labels ]

    lods = []
    for i,j,k in zip(bio_chem,[str('Comparison ART Corpus') for i in range(0,len(bio_chem))],biochem_labels):
        lods.append({'Reading_Level':i,'Origin':j,'Web_Link':k})
    df0 = pd.DataFrame(lods)
    return df0,bio_chem,biochem_labels

def distribution_plot_from_scrape(ar,author_name,scraped_labels,standard_sci,df0):
    ar = [ t for t in ar if t['standard']<45 ]
    group_labels = ['Author: '+str(author_name)]#, 'Group 2', 'Group 3']
    lods = []
    for i,j,k in zip(standard_sci,[str(author_name) for i in range(0,len(ar))],scraped_labels):
        lods.append({'Reading_Level':i,'Origin':j,'Web_Link':k})
    df1 = pd.DataFrame(lods)
    df1.drop_duplicates(subset = "Web_Link", inplace = True)
    df = pd.concat([df1,df0])
    fig = px.histogram(df, x="Reading_Level", y="Web_Link", color="Origin",
                    marginal="box",
                    opacity=0.7,# marginal='violin',# or violin, rug
                    hover_data=df.columns,
                    hover_name=df["Web_Link"],
                    color_discrete_sequence=colors, 
                    histfunc='count',
                    orientation='v',
                    nbins=NBINS
                    )

    fig.update_layout(title_text='Scraped author {0} versus ART Corpus'.format(author_name)
                                ,width=900, height=900)
    return df1,fig

def grand_distribution_plot(ar,scraped_labels,standard_sci,df0,author_name = ''):
    #standard_sci = [ t['standard'] for t in ar ]
    #scraped_labels = [ str(x['link']) for x in ar]
    group_labels = ['Author Scraped']#, 'Group 2', 'Group 3']
    lods = []
    for i,j,k in zip(standard_sci,[str(author_name) for i in range(0,len(ar))],scraped_labels):
        lods.append({'Reading_Level':i,'Origin':j,'Web_Link':k})
    df1 = pd.DataFrame(lods)
    df1.drop_duplicates(subset = "Web_Link", inplace = True)
    df = pd.concat([df1,df0])
    x1 = df0['Reading_Level']
    x2 = df1['Reading_Level']

    group_labels = ['Comparison Art Corpus ', str(author_name)]

    theme = px.colors.diverging.Portland
    colors = [theme[0], theme[1]]
    colors = [theme[-1], theme[-2]]
    rt=list(pd.Series(scraped_labels))
    fig = ff.create_distplot([x1, x2], group_labels, bin_size=2,colors=colors,rug_text=rt)
    hover_trace = [t for t in fig['data'] if 'text' in t]
    fig.update_layout(title_text='Scraped author versus ART Corpus')
    fig.update_layout(width=900, height=600)#, hovermode='x')
    return df1,fig
def push_frame_to_screen(labels,readability_vector):
    df_links = pd.DataFrame()
    df_links['Web_Link'] = pd.Series(labels)
    df_links['Reading_Level'] = pd.Series(readability_vector)
    df_links.drop_duplicates(subset = "Web_Link", inplace = True)
    df_links['Web_Link'] = df_links['Web_Link'].apply(make_clickable)
    df_links = df_links.to_html(escape=False)
    st.write(df_links, unsafe_allow_html=True)
def art_cloud_wl(acorpus):
    wc = WordCloud()
    wc.generate_from_lengths = MethodType(generate_from_lengths,wc)
    fig = plt.figure()
    
    tokens = word_tokenize(acorpus)

    wordcloud = wc.generate_from_lengths(tokens)
    #wc = WordCloud().generate_from_frequencies(frequencies=di)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    '''
    ### A second word cloud where word length controls word size, not word frequency in text
    '''

    st.pyplot()
'''
TOUR = False
if TOUR:
    NAME1 = args.author1
    online_app_backend.call_from_front_end(NAME,NAME1=author2,tour=TOUR,anon=anon,verbose=verbose)
else:
    NAME = "S S Phatak"
    verbose = False
    online_app_backend.call_from_front_end(NAME,verbose=verbose)
'''