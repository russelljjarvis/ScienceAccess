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
import nltk

import streamlit as st



from science_access.online_app_backend import call_from_front_end
from science_access.online_app_backend import ar_manipulation

trainingDats = pickle.load(open('data/trainingDats.p','rb'))
bio_chem = [ t['sp'] for t in trainingDats ]

biochem_labels =  [ x['file_name'] for x in trainingDats if 'file_name' in x.keys()]
biochem_labels = [x.split("/")[-1] for x in biochem_labels ]

lods = []
for i,j,k in zip(bio_chem,[str('Comparison Data') for i in range(0,len(bio_chem))],biochem_labels):
     lods.append({'Sentiment':i,'Origin':j,'Web_Link':k})
df0 = pd.DataFrame(lods)

theme = px.colors.diverging.Portland
colors = [theme[0], theme[1]]
st.title('Search Sentiment Polarity of Academic')
author_name = st.text_input('Enter Author:')
def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    text = link#.split('=')[1]
    return f'<a target="_blank" href="{link}">{text}</a>'

#if 'DYNO' in os.environ:
heroku = True
#else:
#    heroku = False
#st.text(str('heroku')+str(heroku))
with open('data/_author_specificSayali Phatak.p','rb') as f: 
    contents = pickle.load(f)   
(NAME,ar,df,datay,scholar_link) =  contents     
cached_author_name = "Sayali Phatak"

if author_name:
    ar = call_from_front_end(author_name)
    # remove false outliers.
    ar = [ t for t in ar if t['sp']<45 ]

    standard_sci = [ t['sp'] for t in ar ]
    group_labels = ['Author: '+str(author_name)]#, 'Group 2', 'Group 3']
    scraped_labels = [ str(x['link']) for x in ar]


    lods = []
    for i,j,k in zip(standard_sci,[str(author_name) for i in range(0,len(ar))],scraped_labels):
        lods.append({'Sentiment':i,'Origin':j,'Web_Link':k})
    df1 = pd.DataFrame(lods)
    df = pd.concat([df1,df0])
    if not heroku:

        fig0 = px.histogram(df, x="Sentiment", y="Web_Link", color="Origin",
                        marginal="box",
                        opacity=0.7,# marginal='violin',# or violin, rug
                        hover_data=df.columns,
                        hover_name=df["Web_Link"],
                        color_discrete_sequence=colors)

        fig0.update_layout(title_text='Scholar scraped {0} Versus Art Corpus'.format(author_name),width=900, height=900)#, hovermode='x')
                
        st.write(fig0)
        cached = False
    else:
        df_links = pd.DataFrame()
        df_links['Web_Link'] = pd.Series(scraped_labels)
        df_links['Sentiment'] = pd.Series(standard_sci)
        df_links['Web_Link'] = df_links['Web_Link'].apply(make_clickable)
        df_links = df_links.to_html(escape=False)
        st.write(df_links, unsafe_allow_html=True)

        x1 = df0['Sentiment']#np.random.randn(200)
        x2 = df1['Sentiment']#np.random.randn(200) + 2
        if author_name:
            group_labels = ['Comparison Data ', str(author_name)]
        else:
            group_labels = ['Comparison Data ', str(cached_author_name)]
        colors = [theme[-1], theme[-2]]
        rt=list(pd.Series(scraped_labels))
        fig = ff.create_distplot([x1, x2], group_labels, bin_size=2,colors=colors,rug_text=rt)
        hover_trace = [t for t in fig['data'] if 'text' in t]
        fig.update_layout(title_text='Scholar scraped Author Versus Art Corpus')
        fig.update_layout(width=900, height=600)#, hovermode='x')
        '''
    	Displaying stored results until a new author search is entered.
    	'''
        st.write(fig) 




else:      
    cached = True


    (ar, trainingDats) = ar_manipulation(ar)
    standard_sci = [ t['sp'] for t in ar ]

    scraped_labels = [ str(x['link']) for x in ar]
    group_labels = ['Author Scraped']#, 'Group 2', 'Group 3']
    lods = []
    for i,j,k in zip(standard_sci,[str(cached_author_name) for i in range(0,len(ar))],scraped_labels):
        lods.append({'Sentiment':i,'Origin':j,'Web_Link':k})
    df1 = pd.DataFrame(lods)
    df = pd.concat([df1,df0])
    if not heroku:

        fig = px.histogram(df, y="Web_Link", x="Sentiment", color="Origin",
                        marginal="box",
                        opacity=0.7,
                        hover_data=df.columns,
                        hover_name=df["Web_Link"],
                        color_discrete_sequence=colors)

        fig.update_layout(title_text='Scholar {0} Versus Art Corpus'.format(cached_author_name),width=900, height=600)
        '''
    	Displaying stored results until a new author search is entered.
    	'''
        st.write(fig)

    else:
        df_links = pd.DataFrame()
        df_links['Web_Link'] = pd.Series(scraped_labels)
        df_links['Sentiment'] = pd.Series(standard_sci)
        df_links['Web_Link'] = df_links['Web_Link'].apply(make_clickable)
        df_links = df_links.to_html(escape=False)
        st.write(df_links, unsafe_allow_html=True)

        x1 = df0['Sentiment']#np.random.randn(200)
        x2 = df1['Sentiment']#np.random.randn(200) + 2
        if author_name:
            group_labels = ['Comparison Data ', str(author_name)]
        else:
            group_labels = ['Comparison Data ', str(cached_author_name)]
        colors = [theme[-1], theme[-2]]
        rt=list(pd.Series(scraped_labels))
        fig = ff.create_distplot([x1, x2], group_labels, bin_size=2,colors=colors,rug_text=rt)
        hover_trace = [t for t in fig['data'] if 'text' in t]
        fig.update_layout(title_text='Scholar scraped Author Versus Art Corpus')
        fig.update_layout(width=900, height=600)#, hovermode='x')
        '''
    	Displaying stored results until a new author search is entered.
    	'''
        st.write(fig) 

if cached:

    '''

    ### Total number {0} scraped documents:

    '''.format('previously')
    st.text(len(ar))
else:
    '''

    ### Total number {0} scraped documents:

    '''.format('previously')
    st.text(len(ar))
if np.mean(standard_sci) < np.mean(bio_chem):
    '''


    ### This author was on average easier to read as the average of ARTCORPUS:
    A varied collection of biochemistry science papers
    '''

if np.mean(standard_sci) >= np.mean(bio_chem):
    '''


    ### This author was on average harder or just as hard to read as average of ARTCORPUS:
    A varied collection of biochemistry science papers
    '''


sci_corpus = ''

black_list = ['et', 'al','text','crossref','cross', 'ref','google','scholar', 'article','pubmed','full']

for t in ar:
    if 'tokens' in t.keys():
        for s in t['tokens']:
            if s not in set(black_list):
                sci_corpus+=str(' ')+s


def art_cloud(acorpus):

    # Generate a word cloud image

    wordcloud = WordCloud().generate(acorpus)
    fig = plt.figure()

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot()

#DEVELOP = True
#if DEVELOP:
import scipy
twosample_results = scipy.stats.ttest_ind(bio_chem, standard_sci)

matrix_twosample = [
    ['', 'Test Statistic', 'p-value'],
    ['Sample Data', twosample_results[0], twosample_results[1]]
]

fig = ff.create_table(matrix_twosample, index=True)
'''
###  A t-test 
 it checks if the authors distribution 
 is significantly different from the ART-corpus distribution
'''
st.write(fig)
#py.iplot(twosample_table, filename='twosample-table')
'''
### Links 
to the articles that were used to perform this calculation
'''

df_links = pd.DataFrame()
df_links['Web_Link'] = pd.Series(scraped_labels)
df_links['Sentiment'] = pd.Series(standard_sci)
#st.write(df)
# link is the column with hyperlinks
df_links['Web_Link'] = df_links['Web_Link'].apply(make_clickable)
df_links = df_links.to_html(escape=False)
st.write(df_links, unsafe_allow_html=True)
# Create a list of possible values and multiselect menu with them in it.



'''


### Here are some word clouds, that show the frequency of scraped texts
You can eye ball them to see if they fit your intuition about what your searched author writes about
'''
try:
    art_cloud(sci_corpus)
except:

    pass
if not heroku:
    df_links = pd.DataFrame()
    df_links['Web_Link'] = pd.Series(scraped_labels)
    df_links['Sentiment'] = pd.Series(standard_sci)
    #st.write(df)
    # link is the column with hyperlinks
    df_links['Web_Link'] = df_links['Web_Link'].apply(make_clickable)
    df_links = df_links.to_html(escape=False)
    st.write(df_links, unsafe_allow_html=True)

    x1 = df0['Sentiment']#np.random.randn(200)
    x2 = df1['Sentiment']#np.random.randn(200) + 2
    if author_name:
        group_labels = ['Comparison Data ', str(author_name)]
    else:
        group_labels = ['Comparison Data ', str(cached_author_name)]
    colors = [theme[-1], theme[-2]]
    rt=list(pd.Series(scraped_labels))
    fig = ff.create_distplot([x1, x2], group_labels, bin_size=2,colors=colors,rug_text=rt)
    hover_trace = [t for t in fig['data'] if 'text' in t]
    fig.update_layout(title_text='Scholar scraped Author Versus Art Corpus')
    fig.update_layout(width=900, height=600)#, hovermode='x')
    st.write(fig)

list_df = pickle.load(open("data/benchmarks.p","rb")) 
bm = pd.DataFrame(list_df)

bm = bm.rename(columns={'link': 'Web_Link', 'standard': 'Sentiment'})
bm["Origin"] = pd.Series(["Benchmark" for i in range(0,len(bm))])

bm = bm.drop(4, axis=0)
bm = bm.drop(5, axis=0)

bm_temp = pd.DataFrame()
bm_temp["Origin"] = bm["Origin"]
bm_temp["Web_Link"] = bm["Web_Link"]
bm_temp["Sentiment"] = bm["sp"]
bm = copy.copy(bm_temp)

bm_temp['Web_Link'] = bm_temp['Web_Link'].apply(make_clickable)
bm_temp = bm_temp.to_html(escape=False)

'''
In the table below there are benchmarks texts that are 
used as a comparison to investigate some very easy to read scientific writing.
and some very cryptic and unreadable texts too.
'''

st.write(bm_temp, unsafe_allow_html=True)

x1 = bm['Sentiment']
x2 = df1['Sentiment']

x3 = df0['Sentiment']


rt=list(bm['Web_Link'])
rt.extend(list(df1['Web_Link']))
rt.extend(list(df0['Web_Link']))

colors = [theme[0], theme[4],theme[2]]
if author_name:
    group_labels = ['Ideal Bench Marks ', str(author_name), str('Comparison Data')]
else:
    group_labels = ['Ideal Bench Marks  ', str(cached_author_name), str('Comparison Data')]

fig = ff.create_distplot([x1, x2, x3], group_labels, bin_size=1,colors=colors,rug_text=rt)

hover_trace = [t for t in fig['data'] if 'text' in t]

fig.update_layout(title_text='Benchmarks versus scraped Author')
fig.update_layout(width=900, height=600)#, hovermode='x')

st.write(fig)
import random
def display_closestwords_tsnescatterplot(model, size):

    word = random.choice(sci_corpus)
    arr = np.empty((0,size), dtype='f')
    word_labels = [word]
    close_words = model.similar_by_word(word)
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
    fig = plt.figure()
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()
    
    #plt.imshow(wordcloud, interpolation='bilinear')
    #plt.axis("off")
    st.pyplot()
from gensim.models import Word2Vec
model = Word2Vec(sci_corpus, min_count=1,size= 50,workers=3, window =3, sg = 1)
display_closestwords_tsnescatterplot(model, 10)
#ARTCORPUS = pickle.load(open('traingDats.p','rb'))
#acorpus = ''
#for t in ARTCORPUS:
#    if 'tokens' in t.keys():
#        for s in t['tokens']:
#            acorpus+=str(' ')+s