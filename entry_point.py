import streamlit as st



from online_app_backend import call_from_front_end
from online_app_backend import ar_manipulation
import pandas as pd
import pickle
import numpy as np
import plotly.figure_factory as ff
import os
import plotly.express as px
from plotly.subplots import make_subplots

import nltk
try:
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
except:
    nltk.download('punkt')
    nltk.download('stopwords')

if not(os.path.exists('traingDats.p?dl=0') or os.path.exists('traingDats.p')):

    os.system('wget https://www.dropbox.com/s/3h12l5y2pn49c80/traingDats.p?dl=0')
    os.system('wget https://www.dropbox.com/s/crarli3772rf3lj/more_authors_results.p?dl=0')
    os.system('wget https://www.dropbox.com/s/x66zf52himmp5ox/benchmarks.p?dl=0')

if os.path.exists("traingDats.p?dl=0") and not os.path.exists("traingDats.p"):
    os.system('mv traingDats.p?dl=0 traingDats.p')
    os.system('mv benchmarks.p?dl=0 benchmarks.p')
    os.system('mv more_authors_results.p?dl=0 more_authors_results.p')


trainingDats = pickle.load(open('traingDats.p','rb'))
bio_chem = [ t['standard'] for t in trainingDats ]
biochem_labels =  [ x['file_name'] for x in trainingDats if 'file_name' in x.keys()]
biochem_labels = [x.split("/")[-1] for x in biochem_labels ]

lods = []
for i,j,k in zip(bio_chem,[str('Comparison Data') for i in range(0,len(bio_chem))],biochem_labels):
     lods.append({'Reading_Level':i,'Origin':j,'Web_Link':k})
df0 = pd.DataFrame(lods)

theme = px.colors.diverging.Portland
colors = [theme[0], theme[1]]
st.title('Search Reading Difficulty of Academic Author')
author_name = st.text_input('Enter Author:')
if author_name:
    ar = call_from_front_end(author_name)
    standard_sci = [ t['standard'] for t in ar ]
    group_labels = ['Author: '+str(author_name)]#, 'Group 2', 'Group 3']
    scraped_labels = [ str(x['link']) for x in ar]


    lods = []
    for i,j,k in zip(standard_sci,[str(author_name) for i in range(0,len(ar))],scraped_labels):
        lods.append({'Reading_Level':i,'Origin':j,'Web_Link':k})
    df1 = pd.DataFrame(lods)
    df = pd.concat([df1,df0])

    #fig0 = px.histogram(df, x="Reading_Level", y="Web_Link", color="Origin",
    #                marginal="rug",# marginal='violin',# or violin, rug
    #                hover_data=df.columns)
    fig0 = px.histogram(df, x="Reading_Level", y="Web_Link", color="Origin",
                    marginal="violin",
                    opacity=0.7,# marginal='violin',# or violin, rug
                    hover_data=df.columns, color_discrete_sequence=colors)

    fig0.update_layout(title_text='Scholar scraped {0} Versus Art Corpus'.format(author_name),width=900, height=900)#, hovermode='x')
            
    st.write(fig0)


else:   
    try:
        with open('_author_specificSayali Phatak.p','rb') as f: contents = pickle.load(f)   
    except:
        with open('_author_specificDavid Grayden.p','rb') as f: contents = pickle.load(f)
    (NAME,ar,df,datay,scholar_link) =  contents     
    (ar, trainingDats) = ar_manipulation(ar)
    standard_sci = [ t['standard'] for t in ar ]

    scraped_labels = [ str(x['link']) for x in ar]
    group_labels = ['Author Scraped']#, 'Group 2', 'Group 3']
    #colors = ['#393E46', '#2BCDC1', '#F66095']

    #fig = ff.create_distplot([standard_sci], group_labels, colors=colors,
    #                         bin_size=[0.3, 0.2, 0.1], show_curve=True)

    lods = []
    for i,j,k in zip(standard_sci,[str('S Phatak') for i in range(0,len(ar))],scraped_labels):
        lods.append({'Reading_Level':i,'Origin':j,'Web_Link':k})
    df1 = pd.DataFrame(lods)
    df = pd.concat([df1,df0])
    
    #colors = [colors[0], colors[1]]

    fig0 = px.histogram(df, x="Reading_Level", y="Web_Link", color="Origin",
                    marginal="rug",
                    opacity=0.7,# marginal='violin',# or violin, rug
                    hover_data=df.columns,
                    color_discrete_sequence=colors)

    fig0.update_layout(title_text='Scholar S Phatak Versus Art Corpus',width=900, height=600)#, hovermode='x')
            
    st.write(fig0)
'''
### Total number scraped documents:
'''
st.text(len(ar))







x1 = df0['Reading_Level']#np.random.randn(200)
x2 = df1['Reading_Level']#np.random.randn(200) + 2
if author_name:
    group_labels = ['Comparison Data ', str(author_name)]
else:
    group_labels = ['Comparison Data ', str('search_author')]


# Create distplot with curve_type set to 'normal'
colors = [theme[-1], theme[-2]]

rt=list(df['Web_Link'])
#st.text('number scraped documents: {0}'.format(rt))

fig = ff.create_distplot([x1, x2], group_labels, bin_size=2,colors=colors,rug_text=rt)

hover_trace = [t for t in fig['data'] if 'text' in t]

fig.update_layout(title_text='Scholar scraped Author Versus Art Corpus')
fig.update_layout(width=900, height=600)#, hovermode='x')

st.write(fig)

#print(group_labels)
#group_labels = ['Biochemistry Documents']#, 'Group 2', 'Group 3']

#colors = ['#393E46']#, '#2BCDC1', '#F66095']

#fig = ff.create_distplot([standard_sci], group_labels, colors=colors,
#                         bin_size=[0.3, 0.2, 0.1], show_curve=True)

