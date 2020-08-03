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


from science_access.online_app_backend import call_from_front_end
from science_access.online_app_backend import ar_manipulation
from science_access.word_cloud_by_word_len import generate_from_lengths
from science_access.utils import check_passive



trainingDats = pickle.load(open('data/trainingDats.p','rb'))
bio_chem = [ t['standard'] for t in trainingDats ]
biochem_labels =  [ x['file_name'] for x in trainingDats if 'file_name' in x.keys()]
biochem_labels = [x.split("/")[-1] for x in biochem_labels ]

lods = []
for i,j,k in zip(bio_chem,[str('Comparison Data') for i in range(0,len(bio_chem))],biochem_labels):
     lods.append({'Reading_Level':i,'Origin':j,'Web_Link':k})
df0 = pd.DataFrame(lods)

theme = px.colors.diverging.Portland
colors = [theme[0], theme[1]]
st.title('Search Reading Complexity of an Author')
author_name = st.text_input('Enter Author Name:')
#st.title('Optionally add number of documents to use')
#ndocs = st.text_input('Enter Number of documents:, default 8')

def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    text = link#.split('=')[1]
    return f'<a target="_blank" href="{link}">{text}</a>'

USE_OA_DOI = False
with open('data/_author_specificSayali Phatak.p','rb') as f:
    contents = pickle.load(f)
(NAME,ar,df,datay,scholar_link) =  contents
cached_author_name = "Sayali Phatak"
nbins = 40

if author_name:
    ar = call_from_front_end(author_name)
    # remove false outliers.
    ar = [ t for t in ar if t['standard']<45 ]

    standard_sci = [ t['standard'] for t in ar ]
    group_labels = ['Author: '+str(author_name)]#, 'Group 2', 'Group 3']
    scraped_labels = [ str(x['link']) for x in ar]


    lods = []
    for i,j,k in zip(standard_sci,[str(author_name) for i in range(0,len(ar))],scraped_labels):
        lods.append({'Reading_Level':i,'Origin':j,'Web_Link':k})
    df1 = pd.DataFrame(lods)
    df1.drop_duplicates(subset = "Web_Link", inplace = True)

    df = pd.concat([df1,df0])
    if not USE_OA_DOI:


        bin_width= 22

        fig0 = px.histogram(df, x="Reading_Level", y="Web_Link", color="Origin",
                        marginal="box",
                        opacity=0.7,# marginal='violin',# or violin, rug
                        hover_data=df.columns,
                        hover_name=df["Web_Link"],
                        color_discrete_sequence=colors, 
                        histfunc='count',
                        orientation='v',
                        nbins=nbins
						)

        fig0.update_layout(title_text='Scraped author {0} versus ART Corpus'.format(author_name),width=900, height=900)#, hovermode='x')

        st.write(fig0)
        cached = False
    else:
        df_links = pd.DataFrame()
        df_links['Web_Link'] = pd.Series(scraped_labels)
        df_links['Reading_Level'] = pd.Series(standard_sci)

        df_links.drop_duplicates(subset = "Web_Link", inplace = True)

        df_links['Web_Link'] = df_links['Web_Link'].apply(make_clickable)
        df_links = df_links.to_html(escape=False)
        st.write(df_links, unsafe_allow_html=True)

        x1 = df0['Reading_Level']
        x2 = df1['Reading_Level']
        if author_name:
            group_labels = ['Comparison Data ', str(author_name)]
        else:
            group_labels = ['Comparison Data ', str(cached_author_name)]
        colors = [theme[-1], theme[-2]]
        rt=list(pd.Series(scraped_labels))
        fig = ff.create_distplot([x1, x2], group_labels, bin_size=2,colors=colors,rug_text=rt)
        hover_trace = [t for t in fig['data'] if 'text' in t]
        fig.update_layout(title_text='Scraped author versus ART Corpus')
        fig.update_layout(width=900, height=600)
        '''
    	Displaying stored results until a new author search is performed.
    	'''
        st.write(fig)

else:

    cached = True


    (ar, trainingDats) = ar_manipulation(ar)
    standard_sci = [ t['standard'] for t in ar ]

    scraped_labels = [ str(x['link']) for x in ar]
    group_labels = ['Author Scraped']#, 'Group 2', 'Group 3']
    lods = []
    for i,j,k in zip(standard_sci,[str(cached_author_name) for i in range(0,len(ar))],scraped_labels):
        lods.append({'Reading_Level':i,'Origin':j,'Web_Link':k})
    df1 = pd.DataFrame(lods)
    df1.drop_duplicates(subset = "Web_Link", inplace = True)

    df = pd.concat([df1,df0])
    if not USE_OA_DOI:

        fig = px.histogram(df, y="Web_Link", x="Reading_Level", color="Origin",
                        marginal="box",
                        opacity=0.7,
                        hover_data=df.columns,
                        hover_name=df["Web_Link"],
                        color_discrete_sequence=colors,
                        histfunc='count',
                        orientation='v',
                        nbins=nbins)

        fig.update_layout(title_text='Scraped author {0} versus ART Corpus'.format(cached_author_name),width=900, height=600)
        '''
    	Displaying stored results until a new author search is performed.
    	'''
        st.write(fig)

    else:
        df_links = pd.DataFrame()
        df_links['Web_Link'] = pd.Series(scraped_labels)
        df_links['Reading_Level'] = pd.Series(standard_sci)

        df_links.drop_duplicates(subset = "Web_Link", inplace = True)

        df_links['Web_Link'] = df_links['Web_Link'].apply(make_clickable)
        df_links = df_links.to_html(escape=False)
        st.write(df_links, unsafe_allow_html=True)

        x1 = df0['Reading_Level']
        x2 = df1['Reading_Level']
        if author_name:
            group_labels = ['Comparison Data ', str(author_name)]
        else:
            group_labels = ['Comparison Data ', str(cached_author_name)]
        colors = [theme[-1], theme[-2]]
        rt=list(pd.Series(scraped_labels))
        fig = ff.create_distplot([x1, x2], group_labels, bin_size=2,colors=colors,rug_text=rt)
        hover_trace = [t for t in fig['data'] if 'text' in t]
        fig.update_layout(title_text='Scraped author versus ART Corpus')
        fig.update_layout(width=900, height=600)#, hovermode='x')
        '''
    	Displaying stored results until a new author search is performed.
    	'''
        st.write(fig)


st.markdown('''

### There were a total number of {0} documents scraped during this query.

'''.format(len(df1))) #format(len(ar))) - changed this to account for duplicates

st.markdown('''

### The average reading level of these documents was {0}.

'''.format(round(np.mean(standard_sci)),3))

st.markdown('')
st.markdown('')


'''
### Links to articles obtained from the search.
'''

df_links = pd.DataFrame()
df_links['Web_Link'] = pd.Series(scraped_labels)
df_links['Reading_Level'] = pd.Series(standard_sci)

df_links.drop_duplicates(subset = "Web_Link", inplace = True)

df_links['Web_Link'] = df_links['Web_Link'].apply(make_clickable)
df_links = df_links.to_html(escape=False)
st.write(df_links, unsafe_allow_html=True)
# Create a list of possible values and multiselect menu with them in it.

'''
Duplicate entries have been removed.
'''

st.markdown('')
st.markdown('')

'''
### These links are identified individually on the histogram below
'''
if not USE_OA_DOI:

    x1 = df0['Reading_Level']
    if author_name:
        group_labels = [str(author_name)]
    else:
        group_labels = [str(cached_author_name)]
    colors = [theme[-1], theme[-2]]
    rt=list(pd.Series(scraped_labels))
    fig = ff.create_distplot([x1], group_labels, bin_size=1,colors=colors,rug_text=rt)
    hover_trace = [t for t in fig['data'] if 'text' in t]
    fig.update_layout(title_text='')
    fig.update_layout(width=900, height=600)#, hovermode='x')
    st.write(fig)


st.markdown('''
### The average reading level of the scraped work was {0}. For comparison, average adult reads at at 8th grade reading level'''.format(round(np.mean(standard_sci)),3))


'''
Here are a few additional established text sources of known complexity for comparison.
'''

'''
| Text Source | Mean Complexity | Description |
|----------|----------|:-------------:|
| Upgoer 5                            | 6   | library using only the 10,000 most commonly occurring English words |
| Wikipedia                               | 14.9 | free, popular, crowdsourced encyclopedia   |
| Post-Modern Essay Generator (PMEG)  | 16.5 | generates output consisting of sentences that obey the rules of written English, but without restraints on the semantic conceptual references   |
| Art Corpus                       | 18.68  | library of scientific papers published in The Royal Society of Chemistry |
'''

st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')


'''
### Word cloud based on the scraped texts
'''

sci_corpus = ''

# todo, make blacklist importable and same as in utils.
black_list = ['et', 'al','text','crossref','isigoogle',
              'cross', 'ref','google','scholar',
              'article','pubmed','full','doi','org','http',
              'copyright', 'org','figure','pubmed','accessshoping']


def create_giant_strings(ar):
    sci_corpus = ''
    for t in ar:
        if 'tokens' in t.keys():
            for s in t['tokens']:
                if s not in black_list:
                    if "." in s:
                        temp = s.split(".")#, " ")
                        sci_corpus+=str(' ')+temp[0]
                        sci_corpus+=str(' ')+temp[1]
                    if s not in set(black_list):
                        sci_corpus+=str(' ')+s#+str(' ')
    return sci_corpus
sci_corpus = create_giant_strings(ar)
bio_corpus = create_giant_strings(trainingDats)

def art_cloud(acorpus):

    # Generate a word cloud image
    WC = WordCloud(background_color="white")


    
    fig = plt.figure()
    plt.imshow(wordcloud, interpolation="nearest", aspect="auto")
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot()
    plt.show()


try:
    art_cloud(sci_corpus)
except:
    pass

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




def art_cloud_wl(acorpus):
    wc = WordCloud()
    wc.generate_from_lengths = MethodType(generate_from_lengths,wc)

    fig = plt.figure()
    from nltk import word_tokenize

    tokens = word_tokenize(acorpus)

    wordcloud = wc.generate_from_lengths(tokens)
    #wc = WordCloud().generate_from_frequencies(frequencies=di)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    '''
    ### A second word cloud where word length controls word size, not word frequency in text
    '''

    st.pyplot()


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



#list_df = pickle.load(open("data/benchmarks.p","rb"))
#bm = pd.DataFrame(list_df)

#bm = bm.rename(columns={'link': 'Web_Link', 'standard': 'Reading_Level'})
#bm["Origin"] = pd.Series(["Benchmark" for i in range(0,len(bm))])

#bm = bm.drop(4, axis=0)
#bm = bm.drop(5, axis=0)

#bm_temp = pd.DataFrame()
#bm_temp["Origin"] = bm["Origin"]
#bm_temp["Web_Link"] = bm["Web_Link"]
#bm_temp["Reading_Level"] = bm["Reading_Level"]
#bm = copy.copy(bm_temp)

#bm_temp['Web_Link'] = bm_temp['Web_Link'].apply(make_clickable)
#bm_temp = bm_temp.to_html(escape=False)

#'''
#In the table below there are a few established
#benchmark texts for some very easy to read scientific writing (0)
#and some very cryptic and unreadable texts (3).
#These established texts are shown relative to the entered author's work
#'''

#st.write(bm_temp, unsafe_allow_html=True)

#x1 = bm['Reading_Level']
#x2 = df1['Reading_Level']

#x3 = df0['Reading_Level']


#rt=list(bm['Web_Link'])
#rt.extend(list(df1['Web_Link']))
#rt.extend(list(df0['Web_Link']))

#colors = [theme[0], theme[4],theme[2]]
#if author_name:
#    group_labels = ['Ideal Bench Marks ', str(author_name), str('Comparison Data')]
#else:
#    group_labels = ['Ideal Bench Marks  ', str(cached_author_name), str('Comparison Data')]

#fig = ff.create_distplot([x1, x2, x3], group_labels, bin_size=1,colors=colors,rug_text=rt)

#hover_trace = [t for t in fig['data'] if 'text' in t]

#fig.update_layout(title_text='Benchmarks versus scraped Author')
#fig.update_layout(width=900, height=600)#, hovermode='x')

#st.write(fig)

st.markdown('')
st.markdown('')

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
st.markdown("""
# Sentiment:
It is {0} tht the mean sentiment polarity this author is more upbeat than that of the average ARTCORPUS article:
""".format(np.mean(sentiment)<np.mean([r['sp'] for r in ar])))

st.markdown("""
# Uniqueness of words:
It is {0} that the mean uniqueness/ratio of the words used in the ARTCORPUS, this gives an idea of 
how boring or alternatively colorful each article was to read
""".format(np.mean(uniqueness)<np.mean([r['uniqueness'] for r in ar])))


st.markdown(""" # Benchmarks in detail""")

st.markdown("""
1.  Upgoer5 - a library using only the 10,000 most commonly occurring English words[2].
2.  Wikipedia - a free, popular, crowdsourced encyclopedia that is generated from self-nominating volunteers. 
3.  Postmodern Essay Generator (PMEG) - generates output consisting of sentences that obey the rules of written English, but without restraints on the semantic conceptual references [5].
4.  ART Corpus - a library of scientific papers published in The Royal Society of Chemistry (RSC) [1].
""")    
st.markdown("""
| Text Source | Mean Complexity | Unique Words |
|----------|----------|:-------------:|
| [Upgoer 5]()                                     | 7                               | 35,103 |
| Wikipedia                                    | 14.9                         | -  |
| [Post-Modern Essay Generator]() | 16.5                          | -  |
| [Art Corpus]()                                 | 18.68                        | 2,594 |
""",unsafe_allow_html=True) 
"""
#### Here is a source on adult literacy:
"""
"""
Kutner M, Greenberg E, Baer J. National Assessment of Adult Literacy (NAAL): A First Look at the Literacy of America’s Adults in the 21st Century (NCES 2006-470). Washington, DC: National Center for Education Statistics; 2005. http://nces.ed.gov/naal/pdf/2006470.pdf.
"""

#ARTCORPUS = pickle.load(open('traingDats.p','rb'))
#acorpus = ''
#for t in ARTCORPUS:
#    if 'tokens' in t.keys():
#        for s in t['tokens']:
#            acorpus+=str(' ')+s

