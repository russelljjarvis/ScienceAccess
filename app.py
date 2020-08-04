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
#from types import MethodType
#from nltk import word_tokenize


from science_access.t_analysis import not_want_list
from science_access.online_app_backend import call_from_front_end
from science_access.online_app_backend import ar_manipulation
from science_access.word_cloud_by_word_len import generate_from_lengths
from science_access.utils import check_passive

from science_access.enter_author_name import art_cloud, create_giant_strings
from science_access.enter_author_name import distribution_plot_from_scrape, grand_distribution_plot
from science_access.enter_author_name import push_frame_to_screen, fast_art_cloud

trainingDats = pickle.load(open('data/trainingDats.p','rb'))
bio_chem = [ t['standard'] for t in trainingDats ]
biochem_labels =  [ x['file_name'] for x in trainingDats if 'file_name' in x.keys()]
biochem_labels = [x.split("/")[-1] for x in biochem_labels ]

lods = []
for i,j,k in zip(bio_chem,[str('Comparison Data') for i in range(0,len(bio_chem))],biochem_labels):
     lods.append({'Reading_Level':i,'Origin':j,'Web_Link':k})
df0 = pd.DataFrame(lods)

st.title('Search Reading Complexity of an Author')
author_name = st.text_input('Enter Author Name:')
#st.title('Optionally add number of documents to use')
#ndocs = st.text_input('Enter Number of documents:, default 8')


with open('data/_author_specificSayali Phatak.p','rb') as f:
    contents = pickle.load(f)
(NAME,ar,df,datay,scholar_link) =  contents
cached_author_name = "Sayali Phatak"
NBINS = 40

def frame_to_lists(ar):
    scraped_labels = [ str(x['link']) for x in ar]
    standard_sci = [ t['standard'] for t in ar ]
    return scraped_labels, standard_sci

if author_name:
    ar = call_from_front_end(author_name)
    df1,fig = distribution_plot_from_scrape(ar,author_name)
    st.write(fig)
    cached = False
else:
    cached = True
    (ar, trainingDats) = ar_manipulation(ar)

    '''
    Displaying stored results until a new author search is performed.
    '''
    scraped_labels, standard_sci = frame_to_lists(ar)
    push_frame_to_screen(scraped_labels,standard_sci)
                                #ar,scraped_labels,standard_sci,df0,author_name
    df1,fig = grand_distribution_plot(ar,scraped_labels,standard_sci,df0,author_name = cached_author_name)
    st.write(fig)


st.markdown('''

### There were a total number of {0} documents scraped during this query.

'''.format(len(df1))) # - changed this to account for duplicates

st.markdown('''

### The average reading level of these documents was {0}.

'''.format(round(np.mean(standard_sci)),3))

st.markdown('\n\n')


'''
### Links to articles obtained from the search.
'''

push_frame_to_screen(scraped_labels,standard_sci)

# Create a list of possible values and multiselect menu with them in it.



st.markdown('')
st.markdown('')

'''
### These links are identified individually on the histogram below
'''

st.markdown('''
### The average reading level of the scraped work was {0}. For comparison, average adult reads at at 8th grade reading level'''.format(round(np.mean(standard_sci)),3))


'''
Here are a few additional established text sources of known complexity for comparison.
'''

st.markdown(""" # Benchmarks in detail""")

st.markdown("""
1.  [Upgoer 5](https://splasho.com/upgoer5/library.php) - a library using only the 10,000 most commonly occurring English words[2].
2.  Wikipedia - a free, popular, crowdsourced encyclopedia that is generated from self-nominating volunteers. 
Me different wikipedia pages were scraped using the terms: "Vaccine, Climate Change, Genetically Modified Ingredient"
3.  [Post-Modern Essay Generator](http://www.elsewhere.org/journal/pomo/) (PMEG) - generates output consisting of sentences that obey the rules of written English, but without restraints on the semantic conceptual references [5].
4.  [Art Corpus](https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/) - a library of scientific papers published in The Royal Society of Chemistry (RSC) [1].
""")    
st.markdown("""
| Text Source | Mean Complexity | Unique Words |
|----------|----------|:-------------:|
| [Upgoer 5](https://splasho.com/upgoer5/library.php)                                     | 7                               | 35,103 |
| Wikipedia                                    | 14.9                         | -  |
| [Post-Modern Essay Generator](http://www.elsewhere.org/journal/pomo/) | 16.5                          | -  |
| [Art Corpus](https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/)                                 | 18.68                        | 2,594 |
""") 

st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')





if np.mean(standard_sci) < np.mean(bio_chem):

    '''
    ### This author was on average easier to read relative to ART Corpus.

    '''

if np.mean(standard_sci) >= np.mean(bio_chem):


    '''
    ### This author was on average more difficult to read relative to ART Corpus.

    '''







twosample_results = scipy.stats.ttest_ind(bio_chem, standard_sci)

matrix_twosample = [
    ['', 'Test Statistic', 'p-value'],
    ['Result', twosample_results[0], twosample_results[1]]
]

fig = ff.create_table(matrix_twosample, index=True)


'''
A t-test was used to determine whether the reading level for the entered author was
significantly different from that of ART Corpus.
'''
st.write(fig)

if twosample_results[1] >= .05:

    '''
    The reading complexity of the scraped author's text was not significantly different than that of ART Corpus.

    '''
if twosample_results[1] < .05:

    '''
    The reading complexity of the scraped author's text was significantly different than that of ART Corpus.

    '''

st.markdown('\n\n')

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
#### Here is a source on adult literacy:
"""
"""
Kutner M, Greenberg E, Baer J. National Assessment of Adult Literacy (NAAL): A First Look at the Literacy of America’s Adults in the 21st Century (NCES 2006-470). Washington, DC: National Center for Education Statistics; 2005. http://nces.ed.gov/naal/pdf/2006470.pdf.
"""



sci_corpus = create_giant_strings(ar,not_want_list)
bio_corpus = create_giant_strings(trainingDats,not_want_list)

'''
### Word cloud based on the scraped texts
'''
fast_art_cloud(sci_corpus)

st.markdown('')
st.markdown('')


st.markdown(""" # Benchmarks in detail""")

st.markdown("""
1.  [Upgoer 5](https://splasho.com/upgoer5/library.php) - a library using only the 10,000 most commonly occurring English words[2].
2.  Wikipedia - a free, popular, crowdsourced encyclopedia that is generated from self-nominating volunteers. 
3.  Postmodern Essay Generator (PMEG) - generates output consisting of sentences that obey the rules of written English, but without restraints on the semantic conceptual references [5].
4.  ART Corpus - a library of scientific papers published in The Royal Society of Chemistry (RSC) [1].
""")    
st.markdown("""
| Text Source | Mean Complexity | Unique Words |
|----------|----------|:-------------:|
| [Upgoer 5](https://splasho.com/upgoer5/library.php)                                     | 7                               | 35,103 |
| Wikipedia                                    | 14.9                         | -  |
| [Post-Modern Essay Generator](http://www.elsewhere.org/journal/pomo/) | 16.5                          | -  |
| [Art Corpus](https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/)                                 | 18.68                        | 2,594 |
""") 
"""
#### Here is a source on adult literacy:
"""
"""
Kutner M, Greenberg E, Baer J. National Assessment of Adult Literacy (NAAL): A First Look at the Literacy of America’s Adults in the 21st Century (NCES 2006-470). Washington, DC: National Center for Education Statistics; 2005. http://nces.ed.gov/naal/pdf/2006470.pdf.
"""

sentiment=[]
uniqueness=[]
for block in trainingDats:
    uniqueness.append(block['uniqueness'])
    sentiment.append(block['sp'])
st.markdown("""
# Sentiment:
It is {0} tht the mean sentiment polarity this author is more upbeat than that of the average ARTCORPUS article:
""".format(np.mean(sentiment)<np.mean([r['sp'] for r in ar])))


labels = ['This Author positive sentiment','ART Corpus positive sentiment']
values = [np.mean([r['sp'] for r in ar]),np.mean(sentiment)]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
st.write(fig)

st.markdown("""
# Uniqueness of words:
It is {0} that the mean uniqueness/ratio of the words used in the ARTCORPUS, this gives an idea of 
how boring or alternatively colorful each article was to read
""".format(np.mean(uniqueness)<np.mean([r['uniqueness'] for r in ar])))


labels = ['This Author unique words ratio','ART Corpus unique words ratio']
values = [np.mean([ r['uniqueness'] for r in ar]),np.mean(uniqueness)]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
st.write(fig)

