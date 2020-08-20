"""
## Science Readability Project

To ensure that writing is accessible to the general population, authors must consider the length of written text, as well as sentence structure, vocabulary, and other language features. While popular magazines, newspapers, and other outlets purposefully cater language for a wide audience, there is a tendency for academic writing to use more complex, jargon-heavy language.

In the age of growing science communication, this tendency for scientists to use more complex language can carry over when writing in more mainstream media, such as blogs and social media. This can make public-facing material difficult to comprehend, undermining efforts to communicate scientific topics to the general public. While readability tools, such as Readable and Upgoer5 currently exist to report on readability of text, they report the complexity of only a single document. In addition, these tools do not focus on complexity in a more academic-type context.

To address this, we created a tool that uses a data-driven approach to provide authors with insights into the readability of the entirety of their published scholarly work with regard to other text repositories. The tool first quantifies an existing text repository [@Soldatova:2007] with complexity shown to be comparable to that of other scientific journals. The tool subsequently uses this output as a reference to show how the readability of user-selected written work compares to this source.

Ultimately, this tool will expand upon current readability metrics by computing a more detailed and comparative look at the complexity of written text. We hope that this will allow scientists and other experts to better monitor the complexity of their writing relative to other text types, leading to the creation of more accessible online material. And perhaps more broadly contribute to an improved global communication and understanding of complex topics.

Author: [Russell Jarvis](https://github.com/russelljjarvis)\n
Author: [Patrick McGurrin](https://github.com/mcgurrgurr)\n
Source: [Github](https://github.com/russelljjarvis/ScienceAccess)
"""

import streamlit as st
import os
import pandas as pd
import pickle
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import copy
import streamlit as st
import math
import scipy
import plotly.graph_objects as go


from science_access.t_analysis import not_want_list#, 
from science_access.online_app_backend import call_from_front_end
from science_access.online_app_backend import ar_manipulation


from science_access.enter_author_name import art_cloud, create_giant_strings, art_cloud_wl, zipf_plot
from science_access.enter_author_name import distribution_plot_from_scrape, grand_distribution_plot
from science_access.enter_author_name import push_frame_to_screen, fast_art_cloud, grab_data_for_splash
from science_access.enter_author_name import frame_to_lists, try_and_update_cache, get_table_download_link, extra_options


from nltk.corpus import words as english_words


def main():
    with open('data/trainingDats.p','rb') as f:
        trainingDats = pickle.load(f)
        art_frame,bio_chem,biochem_labels = grab_data_for_splash(trainingDats)

    with open('data/_author_specificSayali Phatak.p','rb') as f:
        contents = pickle.load(f)
        (NAME,ar,df,datay,scholar_link) =  contents
    st.title('Search Reading Complexity of an Author')
    author_name = st.text_input('Enter Author Name:')
    radio_value = st.sidebar.radio("Select Search Backend Base Search.net True /or Experimental False",[True,False])
    radio_value = st.sidebar.radio("Target Number of Samples",[10,20,30])
    #st.text(radio_value)
    cached_author_name = "Sayali Phatak"
    NBINS = 40

    if author_name:
        #show_exp = st.checkbox("Include expensive listings")
        #st.markdown("resolving query for author {0}".format(str(author_name)))
        ar = call_from_front_end(author_name,OPENACCESS=radio_value)
        st.text(ar)
        scraped_labels, standard_sci = frame_to_lists(ar)

        push_frame_to_screen(scraped_labels,standard_sci)
        author_specific_frame,fig = grand_distribution_plot(ar,scraped_labels,standard_sci,art_frame,author_name = author_name)

        #author_specific_frame,fig = distribution_plot_from_scrape(ar,author_name,scraped_labels,standard_sci,art_frame)
        st.write(fig)
        cached = False
		# try and update underlying distribution with query, so information about science
		# is culmulative, dynamic.
		# Try to allow researchers of the app to download the data.
		# Via GUI prompts.
		# extra_options(ar,trainingDats,author_specific_frame)
    else:
        cached = True
        author_name = cached_author_name
        (ar, trainingDats) = ar_manipulation(ar)

        '''
        Displaying stored results until a new author search is performed.
        '''
        scraped_labels, standard_sci = frame_to_lists(ar)

        author_specific_frame,fig = grand_distribution_plot(ar,scraped_labels,standard_sci,art_frame,author_name = author_name)
        st.write(fig)
    if not cached:
        st.markdown('''
        ### There were a total number of {0} documents scraped during this query.
        '''.format(len(author_specific_frame))) # - changed this to account for duplicates

    st.markdown('''
    ### The average reading level was {0}.
    '''.format(round(np.mean(standard_sci)),3))

    '''
    In general, we can equate reading level with grade level.
    '''

    st.markdown('\n')

    if np.mean(standard_sci) < np.mean(bio_chem):
        st.markdown('''
        ### {0} was on average easier to read relative to ART Corpus.
        '''.format(author_name))

    if np.mean(standard_sci) >= np.mean(bio_chem):
        st.markdown('''
        ### {0} was on average more difficult to read relative to ART Corpus.
        '''.format(author_name))
		
    st.markdown('-----')
    st.markdown('\n\n')

    '''
    ### Links to articles obtained from the scrape.
    '''

    push_frame_to_screen(scraped_labels,standard_sci)
    # Create a list of possible values and multiselect menu with them in it.
    '''
    Duplicate entries have been removed from analysis.
    '''

    st.markdown('-----')
    st.markdown('\n\n')

    st.markdown('''
    ### The average reading level of the scraped work was {0}.'''.format(round(np.mean(standard_sci)),3))

    '''
    For comparison, [the average adult reads at an 8th grade reading level](http://nces.ed.gov/naal/pdf/2006470.pdf).
    '''

    '''
    Here are a few additional established text sources of known complexity:
    '''

    st.markdown("""
    | Text Source | Mean Complexity | Description |
    |----------|----------|:-------------:|
    | [Upgoer 5](https://splasho.com/upgoer5/library.php)                             | 7     | library using only the 10,000 most commonly occurring English words |
    | [Readability of science declining](https://elifesciences.org/articles/27725)   |  9.0 | example of a scientific article discussing writing to a broad audience in an academic context |
    | [Science of writing](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf) | 14.0 | example of a scientific article discussing writing to a broad audience in an academic context |
    | Wikipedia                                                                       | 14.9   | free, popular, crowdsourced encyclopedia generated from self-nominating volunteers  |
    | [Post-Modern Essay Generator](http://www.elsewhere.org/journal/pomo/)           | 16.5   | generates sentences that obey the rules of written English, but the meaning is nonesense |
    | [Art Corpus](https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/) | 18.68  | library of scientific papers published in The Royal Society of Chemistry |
    """)

    st.markdown('-----')
    st.markdown('\n\n\n\n')


    sci_corpus = create_giant_strings(ar,not_want_list)
    bio_corpus = create_giant_strings(trainingDats,not_want_list)

    '''
    ### Word cloud based on the most common words found in the scraped text
    '''
    fast_art_cloud(sci_corpus)

    st.markdown('-----')
    st.markdown('\n\n')



    """
    ### Word cloud based on the largest words found in the scraped text
    """

    big_words,word_counts_fz = art_cloud_wl(sci_corpus)
    #st.markdown('Here is one of the biggest words: {0}'''.format(str(big_words[0][0])))
    #st.markdown('Here is one of the biggest words: "{0}", you should feed it into PCA of word2vec'.format(str(big_words[0][0])))

    st.markdown('-----')
    st.markdown('\n\n')

    #sci_corpus = create_giant_strings(ar,not_want_list)
	#bio_corpus = create_giant_strings(trainingDats,not_want_list)


    sentiment=[]
    uniqueness=[]
    for block in trainingDats:
        uniqueness.append(block['uniqueness'])
        sentiment.append(block['sp'])
    temp = np.mean(sentiment)<np.mean([r['sp'] for r in ar])
    st.markdown("""
    ### Sentiment
    It is {} that the mean sentiment of {} is more postive relative to that of ART Corpus.
    """.format(temp,author_name))

    temp = '{0} positive sentiment'.format(author_name)
    labels = [temp,'ART Corpus positive sentiment']
    values = [np.mean([r['sp'] for r in ar]),np.mean(sentiment)]

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

if __name__ == "__main__":
    main()

