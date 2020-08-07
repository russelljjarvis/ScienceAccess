import streamlit as st
import os
#import matplotlib.pyplot as plt
#import seaborn as sns
#from wordcloud import WordCloud
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
#from types import MethodType
#from nltk import word_tokenize
import plotly.graph_objects as go


from science_access.t_analysis import not_want_list#, 
from science_access.online_app_backend import call_from_front_end
from science_access.online_app_backend import ar_manipulation
#from science_access.word_cloud_by_word_len import generate_from_lengths

from science_access.enter_author_name import art_cloud, create_giant_strings, art_cloud_wl, zipf_plot
from science_access.enter_author_name import distribution_plot_from_scrape, grand_distribution_plot
from science_access.enter_author_name import push_frame_to_screen, fast_art_cloud, grab_data_for_splash
from science_access.enter_author_name import frame_to_lists, try_and_update_cache, get_table_download_link, extra_options

with open('data/trainingDats.p','rb') as f:
    trainingDats = pickle.load(f)
    df0,bio_chem,biochem_labels = grab_data_for_splash(trainingDats)

st.title('Search Reading Complexity of an Author')
author_name = st.text_input('Enter Author Name:')



with open('data/_author_specificSayali Phatak.p','rb') as f:
    contents = pickle.load(f)
(NAME,ar,df,datay,scholar_link) =  contents
cached_author_name = "Sayali Phatak"
NBINS = 40


if author_name:
    ar = call_from_front_end(author_name)
    scraped_labels, standard_sci = frame_to_lists(ar)

    df1,fig = distribution_plot_from_scrape(ar,author_name,scraped_labels,standard_sci,df0)
    st.write(fig)
    cached = False
    # try and update underlying distribution with query, so information about science
    # is culmulative, dynamic.
    # Try to allow researchers of the app to download the data.
    # Via GUI prompts.
    # extra_options(ar,trainingDats,df1)
else:
    cached = True
    author_name = cached_author_name
    (ar, trainingDats) = ar_manipulation(ar)

    '''
    Displaying stored results until a new author search is performed.
    '''
    scraped_labels, standard_sci = frame_to_lists(ar)
    #push_frame_to_screen(scraped_labels,standard_sci)
    #st.markdown('-----')

    df1,fig = grand_distribution_plot(ar,scraped_labels,standard_sci,df0,author_name = author_name)
    st.write(fig)


st.markdown('''

### There were a total number of {0} documents scraped during this query.

'''.format(len(df1))) # - changed this to account for duplicates

st.markdown('''

### The average reading level of these documents was {0}.

'''.format(round(np.mean(standard_sci)),3))

'''
In general, we can equate reading level with grade level.
'''

st.markdown('-----')
st.markdown('\n\n')

'''
### Links to articles obtained from the search.
'''

push_frame_to_screen(scraped_labels,standard_sci)
# Create a list of possible values and multiselect menu with them in it.

st.markdown('-----')
st.markdown('\n\n')

st.markdown('''
### The average reading level of the scraped work was {0}. For comparison, [the average adult reads at an 8th grade reading level](http://nces.ed.gov/naal/pdf/2006470.pdf)'''.format(round(np.mean(standard_sci)),3))


'''
Here are the metrics for a few additional established text sources of known complexity:
'''

st.markdown("""
| Text Source | Mean Complexity | Description |
|----------|----------|:-------------:|
| [Upgoer 5](https://splasho.com/upgoer5/library.php)                             | 7     | library using only the 10,000 most commonly occurring English words |
| [readability of science declining](https://elifesciences.org/download/aHR0cHM6Ly9jZG4uZWxpZmVzY2llbmNlcy5vcmcvYXJ0aWNsZXMvMjc3MjUvZWxpZmUtMjc3MjUtdjIucGRm/elife-27725-v2.pdf?_hash=WA%2Fey48HnQ4FpVd6bc0xCTZPXjE5ralhFP2TaMBMp1c%3D)   |  9.0 | example of a scientific article discussing writing to a broad audience in an academic context |
| [science of writing](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf) | 14.0 | example of a scientific article discussing writing to a broad audience in an academic context |
| Wikipedia                                                                       | 14.9   | free, popular, crowdsourced encyclopedia generated from self-nominating volunteers  |
| [Post-Modern Essay Generator](http://www.elsewhere.org/journal/pomo/)           | 16.5   | generates output consisting of sentences that obey the rules of written English, but without restraints on the semantic conceptual references   |
| [Art Corpus](https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/) | 18.68  | library of scientific papers published in The Royal Society of Chemistry |
""")

st.markdown('-----')
st.markdown('\n\n\n\n')


sci_corpus = create_giant_strings(ar,not_want_list)
bio_corpus = create_giant_strings(trainingDats,not_want_list)

'''
### Word cloud based on the most common words found in the scraped texts
'''
fast_art_cloud(sci_corpus)

st.markdown('-----')
st.markdown('\n\n')



"""
### Word cloud based on the largest words found in the scraped texts
if message about caching means it will run faster on second run.
"""

big_words,word_counts_fz = art_cloud_wl(sci_corpus)
st.markdown('Here is one of the biggest words found: "{0}")
#st.markdown('Here is one of the biggest words: "{0}", you should feed it into PCA of word2vec'.format(str(big_words[0][0])))

st.markdown('-----')
st.markdown('\n\n')


if np.mean(standard_sci) < np.mean(bio_chem):
    st.markdown('''
    ### {0} was on average easier to read relative to ART Corpus.
    '''.format(author_name))

if np.mean(standard_sci) >= np.mean(bio_chem):
    st.markdown('''
    ### {0} was on average more difficult to read relative to ART Corpus.
    '''.format(author_name))


twosample_results = scipy.stats.ttest_ind(bio_chem, standard_sci)

matrix_twosample = [
    ['', 'Test Statistic', 'p-value'],
    ['Result', twosample_results[0], twosample_results[1]]
]

fig = ff.create_table(matrix_twosample, index=True)

st.markdown('''
A t-test was used to determine whether the reading level for the {0} was
significantly different from that of ART Corpus.
'''.format(author_name))
st.write(fig)

if twosample_results[1] >= .05:
    st.markdown('''
    The reading complexity of {0} text was not significantly different than that of ART Corpus.
    '''.format(author_name))
if twosample_results[1] < .05:
    st.markdown('''
    The reading complexity of {0} text was significantly different than that of ART Corpus.
    '''.format(author_name))


st.markdown('-----')
st.markdown('\n\n')


sentiment=[]
uniqueness=[]
for block in trainingDats:
    uniqueness.append(block['uniqueness'])
    sentiment.append(block['sp'])
temp = np.mean(sentiment)<np.mean([r['sp'] for r in ar])
st.markdown("""
### Sentiment
It is {} that the mean sentiment polarity of {} is more postive relative to that of ART Corpus.
""".format(temp,author_name))

temp = '{0} positive sentiment'.format(author_name)
labels = [temp,'ART Corpus positive sentiment']
values = [np.mean([r['sp'] for r in ar]),np.mean(sentiment)]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
st.write(fig)

st.markdown("""
### Uniqueness of words (different words used / total number of words)
It is {0} that on average the word diversity of {1} is higher relative to that of ART Corpus.
""".format(np.mean(uniqueness)<np.mean([r['uniqueness'] for r in ar]),author_name))


labels = ['{0} unique words ratio'.format(author_name),'ART Corpus unique words ratio']
values = [np.mean([ r['uniqueness'] for r in ar]),np.mean(uniqueness)]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
st.write(fig)


st.markdown('\n\n')
st.markdown('-----')

"""
### Here are some links where you can read about the readability metrics and the algorithms used to compute the metrics:
"""
"""
[Readability Metric Alogrithms and Background](https://en.wikipedia.org/wiki/Readability)
"""
"""
[Gunning Fog Readability Metric Alogrithm](https://en.wikipedia.org/wiki/Gunning_fog_index)
"""

"""
#### [Here is a source](http://nces.ed.gov/naal/pdf/2006470.pdf) about variation in adult literacy:
"""
"""
Kutner M, Greenberg E, Baer J. National Assessment of Adult Literacy (NAAL): A First Look at the Literacy of America’s Adults in the 21st Century (NCES 2006-470). Washington, DC: National Center for Education Statistics; 2005.
"""

st.markdown('-----')
st.markdown('\n\n')

#elaborate_plot(trainingDats)

zipf_plot(word_counts_fz)
#try:
#except:
#    pass
try:
    bio_corpus = create_giant_strings(trainingDats,not_want_list)
    big_words = art_cloud_wl(bio_corpus)
    st.markdown(str(big_words[0][0]))
except:
    pass
