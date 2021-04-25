"""
Summary of application
Here we present a new text analysis tool capable of reviewing the complexity of a scientist’s published work relative to other online-based text repositories. Rather than providing feedback as to the complexity of a single text as previous tools have done, the tool presented here shows the relative complexity across many texts from the same author, while also comparing the readability of the author’s body of work to a variety of other scientific and lay text types.

The goal of this work is to apply a more data-driven approach that provides established academic authors with statistical insights into their body of published peer reviewed work. By monitoring these readability metrics, scientists may be able to cater their writing to reach broader audiences, contributing to an improved global communication and understanding of complex topics.

Author: [Russell Jarvis](https://github.com/russelljjarvis)\n
Author: [Patrick McGurrin](https://github.com/mcgurrgurr)\n
"""

import click
#import argparse
import sys
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
import shelve
import plotly.express as px
import pandas as pd
from random import sample
import click
from typing import List, Any

from science_access.t_analysis import not_want_list
not_want_list.extend(["link","librarian","issue","abstract","science","cookie","publication"])

from science_access.online_app_backend import call_from_front_end
from science_access.online_app_backend import ar_manipulation

# from science_access	import bokeh_word_cloud

from science_access.enter_author_name import (
    art_cloud,
    create_giant_strings,
    art_cloud_wl,
    zipf_plot,
)
from science_access.enter_author_name import (
    distribution_plot_from_scrape,
    grand_distribution_plot,
    data_frames_from_scrape,
)
from science_access.enter_author_name import (
    push_frame_to_screen,
    fast_art_cloud,
    grab_data_for_splash,
)
from science_access.enter_author_name import (
    frame_to_lists,
    try_and_update_cache,
    get_table_download_link,
    extra_options,
)

rd_df = pd.read_csv("Figure4_SourceData1.csv")

rd_df.rename(
    columns={"flesch_fulltexts": "Reading_Level", "journal": "Origin"}, inplace=True
)
rd_df = rd_df[["Reading_Level", "Origin"]]
rd_df["Origin"] = ["ReadabilityScienceDeclining" for i in rd_df["Origin"]]

rd_labels = rd_df["Origin"]
rd_level = rd_df["Reading_Level"]
rd_df = rd_df.loc[sample(list(rd_df.index), 999)]

with open("data/trainingDats.p", "rb") as f:
    trainingDats = pickle.load(f)
    art_df, bio_chem_level, biochem_labels = grab_data_for_splash(trainingDats)
biochem_labels = art_df["Origin"]
bio_chem_level = art_df["Reading_Level"]

#@st.cache(suppress_st_warning=True)
def check_cache(author_name: str,verbose=0):  # ->Union[]
    with shelve.open("fast_graphs_splash.p") as db:
        flag = author_name in db
        if not flag:
            ar = call_from_front_end(author_name)
            scraped_labels, author_score = frame_to_lists(ar)
            ##
            # This shelve
            # caching wont scale on heroku.
            # need TinyDb on Amazon
            ##
            if len(db.keys())<11:
                db[author_name] = {
                    "ar": ar,
                    "scraped_labels": scraped_labels,
                    "author_score": author_score,
                }
        else:
            """
            We have evaluated this query recently, using cached results...
            """

            temp = db[author_name]
            ar = temp["ar"]
            if "standard_sci" in temp.keys():
                author_score = temp["standard_sci"]
            if "author_score" in temp.keys():
                author_score = temp["author_score"]

            scraped_labels = temp["scraped_labels"]

        #experimental = [
        #    np.mean([a["standard_len"], a["ndc"]])
        #    for a in ar
        #    if "standard_len" in a.keys()
        #]
    return ar, author_score, scraped_labels

def show_hardest_passage(ar:List=[])->str:
    largest = 0
    li = 0
    for i,a in enumerate(ar):
        if a["standard"]>largest:
            largest = a["standard"]
            li=i
    if "hard_snippet" in ar[i].keys():
        st.markdown("A hard to read passage from the authors work.")
        if str("can log in with their society credentials") not in ar[i]["hard_snippet"]:
            st.text(ar[i]["hard_snippet"])
    return ar[i]


#def clouds_big_words(sci_corpus):
    #if len(sci_corpus) != 0:

        #st.markdown("-----")
        #st.markdown(""" ### Word Length Word Cloud 	""")
        #st.markdown(
        #    """
		#based on the largest words found in the mined text.
		#These words are likely culprits that hindered readability.
		#"""
        #)
        #big_words, word_counts_fz, fig_wl = art_cloud_wl(sci_corpus)

#@click.command()
#@click.argument('verbose', type=int, default=0)
verbose=0

def main():
    st.title("Search Reading Complexity of an Author")
    author_name = st.text_input("Enter Author Name. Entering a middle initial followed by '.' may improve accuracy of results. Eg. Sayali S. Phatak")

    st.markdown(
        """[Background information for the app](https://github.com/russelljjarvis/ScienceAccess/blob/master/Documentation/paper.md) and
           [source code](https://github.com/russelljjarvis/ScienceAccess)"""
    )

    st.markdown(
        """Note: Search applies [dissmin](https://dissemin.readthedocs.io/en/latest/api.html) API backend"""
    )

    st.markdown("-----")

    if author_name:
    	ar, author_score, scraped_labels = check_cache(author_name,verbose)
    if "ar" in locals():
        df_author, merged_df = data_frames_from_scrape(
            ar, author_name, scraped_labels, author_score, art_df
        )
        #hard=show_hardest_passage(ar)

        st.markdown("-----")
        """
        ## Results:
		"""
        st.markdown("\n\n")

        st.markdown(
            """
		### There was a total of {0} documents mined during this query.
		""".format(
                len(df_author)
            )
        )

        push_frame_to_screen(df_author, scraped_labels)

        # temp = "{0} Summary Readability versus large sample of science".format(author_name)
        # labels = [temp, "ART Corpus readability"]
        # values = [np.mean([r["standard"] for r in ar]), np.mean(bio_chem_level)]
        # fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
        # st.write(fig)

        df_concat_art = pd.concat([rd_df, df_author])
        fig_art = px.box(
            df_concat_art, x="Origin", y="Reading_Level", points="all", color="Origin"
        )
        st.write(fig_art)

        #df_concat_art = pd.concat([art_df, df_author])
        #fig_art = px.box(
        #    df_concat_art, x="Origin", y="Reading_Level", points="all", color="Origin"
        #)
        #st.write(fig_art)

        df0 = df_concat_art

        st.markdown(
            """
		### The average reading level of the mined work was {0}.""".format(
                round(np.mean(author_score)), 3
            )
        )

        """
		For comparison, [the average adult reads at an 8th grade reading level](http://nces.ed.gov/naal/pdf/2006470.pdf).
		"""
        #try:

        st.markdown("\n\n")

        if np.mean(author_score) < np.mean(bio_chem_level):
            st.markdown(
                """
			### The author {0} was on average easier to read relative to the ART Corpus.
			""".format(
                    author_name
                )
            )

        if np.mean(author_score) >= np.mean(bio_chem_level):
            st.markdown(
                """
			### {0} was on average more difficult to read relative to the ART Corpus.
			""".format(
                    author_name
                )
            )

        st.markdown("-----")

        st.markdown(""" ### Word Frequency Word Cloud""")
        """
		The word cloud is based on the most common words found in the mined text.
		This allows the user to validate the text mining work, as the output should match a writers
		own knowledge of concepts in their work.
		"""
        sci_corpus = create_giant_strings(ar, not_want_list)
        fast_art_cloud(sci_corpus)
        #clouds_by_big_words = True
        #if clouds_by_big_words:
        #    clouds_big_words(sci_corpus)


        #if verbose:
        #    st.text(sci_corpus)
        #with shelve.open("fast_graphs_splash.p") as db:
        #    if not author_name in db.keys():
        #        db[author_name] = {
        #            "ar": ar,
        #            "scraped_labels": scraped_labels,
        #            "author_score": author_score,
        #            "sci_corpus": sci_corpus,
        #        }
        st.markdown("\n")

        # sci_corpus = create_giant_strings(ar, not_want_list)
        # bio_corpus = create_giant_strings(trainingDats, not_want_list)

        # st.markdown('Here is one of the biggest words: {0}'''.format(str(big_words[0][0])))
        # st.markdown('Here is one of the biggest words: "{0}", you should feed it into PCA of word2vec'.format(str(big_words[0][0])))

        grab_setr = []
        grab_set1 = []

        for block in trainingDats:
            grab_setr.extend(block["tokens"])
        for block in ar:
            grab_set1.extend(block["tokens"])

        artset = list(grab_setr)
        artset.extend(not_want_list)
        autset = list(set(grab_set1))
        exclusive = [i for i in autset if i not in artset]
        # inclusive = [i for i in autset if i in artset]
        #st.markdown(
        #    "### Concepts that differentiate {0} from other science".format(
        #        author_name
        #    )
        #)
        #exclusive = create_giant_strings(ar, exclusive)

        #fig = fast_art_cloud(exclusive)

        sentiment = []
        uniqueness = []
        for block in trainingDats:
            uniqueness.append(block["uniqueness"])
            sentiment.append(block["sp"])
        temp = np.mean(sentiment) < np.mean([r["sp"] for r in ar])
        if "reading_time" in ar[0].keys():
            average_reading_time = [np.mean([r["reading_time"] for r in ar])]

            st.markdown("""### Reading Time""")
            st.markdown(
                """There were {2} documents. The average reading time
			per document for author {1} was {0} Minutes.
			""".format(
                    np.mean(average_reading_time), author_name, len(ar)
                )
            )

        st.markdown("-----")

        st.markdown("""### Sentiment""")
        st.markdown(
            """It is {} that the mean sentiment of {}'s writing is more postive relative to that of ART Corpus.
					""".format(
                temp, author_name
            )
        )

        temp = "{0} positive sentiment".format(author_name)
        labels = [temp, "ART Corpus positive sentiment"]
        values = [np.mean([r["sp"] for r in ar]), np.mean(sentiment)]

        # urlDat["reading_time"]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
        st.write(fig)

        st.markdown("-----")

        """
		Here are a few established text sources of known complexity.
		Note that in general, we can equate reading level with grade level.
		"""

        st.markdown(
            """
		| Text Source | Mean Complexity | Description |
		|----------|----------|:-------------:|
		| [Upgoer 5](https://splasho.com/upgoer5/library.php)                             | 7     | library using only the 10,000 most commonly occurring English words |
		| [Readability of science declining](https://elifesciences.org/articles/27725)   |  9.0 | example of a scientific article discussing writing to a broad audience in an academic context |
		| [Science of writing](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf) | 14.0 | example of a scientific article discussing writing to a broad audience in an academic context |
		| Wikipedia                                                                       | 14.9   | free, popular, crowdsourced encyclopedia generated from self-nominating volunteers  |
		| [Post-Modern Essay Generator](http://www.elsewhere.org/journal/pomo/)           | 16.5   | generates output consisting of sentences that obey the rules of written English, but without restraints on the semantic conceptual references   |
		| [ART Corpus](https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/) | 18.68  | library of scientific papers published in The Royal Society of Chemistry |
		"""
        )
        st.markdown("-----")

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

        st.markdown("-----")


if __name__ == "__main__":

    main()
