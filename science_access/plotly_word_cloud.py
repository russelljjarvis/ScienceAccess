
from wordcloud import WordCloud, STOPWORDS
#import plotly.graph_objs as go
from bokeh.io import show
from bokeh.models import ColumnDataSource
from bokeh_wordcloud2 import WordCloud2
import streamlit as st
from bokeh.plotting import figure

import random

from bokeh.io import show, curdoc
from bokeh.layouts import widgetbox, column
from bokeh.models import ColumnDataSource, Button, Div

def bokeh_wordcloud(text):
    wc = WordCloud(stopwords = set(STOPWORDS),
                   max_words = 50,
                   max_font_size = 10)
    wc.generate(text)

    word_list=[]
    freq_list=[]
    fontsize_list=[]
    position_list=[]
    orientation_list=[]
    color_list=[]

    for (word, freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)
    '''
    # get the positions
    x=[]
    y=[]
    for i in position_list:
        x.append(i[0])
        y.append(i[1])

    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i*100)
    new_freq_list
    names,weights = zip(*data)
    '''
    test1 = ColumnDataSource({'names':word_list,'weights':freq_list})
    # we will specify just "blue" for the color
    wordcloud = WordCloud2(source=test1,wordCol="names",sizeCol="weights")#,colors="blue")
    '''
    def clicked_word(evt):
        data=test1.data
        if(7 < evt.weight < 20 ):
            new_weight = evt.weight + random.choice([-1,1,1,2])
        elif evt.weight < 7:
            new_weight = evt.weight + random.choice([1,2])
        elif evt.weight < 20:
            new_weight = evt.weight - random.choice([1,2,3])
        weights = list(data['weights'])
        weights[data['names'].index(evt.word)] = new_weight
        test1.data = {'names':data['names'][:],'weights':weights}


    button = Div(text="<h1>Click A Word!</h1>")
    # button.on_click(callback)
    # show(sdp)
    wordcloud.on_click(clicked_word)
    curdoc().add_root(column(wordcloud))
    '''
    st.bokeh_chart(wordcloud)
    return wordcloud

    '''
    trace = go.Scatter(x=x,
                       y=y,
                       textfont = dict(size=fontsize_list,
                                       color=color_list),
                       hoverinfo='text',
                       hovertext=['{0}{1}'.format(w, f) for w, f in zip(word_list, freq_list)],
                       mode='text',
                       text=word_list,
                       text_angle=orientation_list
                      )

    layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                        'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                        'width': 1025,
                        'height': 1025,
                        })
    fig = go.Figure(data=[trace], layout=layout)
    fig["layout"]["width"] = 825
    fig["layout"]["height"] = 825
    '''
