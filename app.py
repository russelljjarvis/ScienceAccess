"""
## Science Readability Project

To ensure that writing is accessible to the general population, authors must consider the length of written text, as well as sentence structure, vocabulary, and other language features. While popular magazines, newspapers, and other outlets purposefully cater language for a wide audience, there is a tendency for academic writing to use more complex, jargon-heavy language.
In the age of growing science communication, this tendency for scientists to use more complex language can carry over when writing in more mainstream media, such as blogs and social media. This can make public-facing material difficult to comprehend, undermining efforts to communicate scientific topics to the general public. While readability tools, such as Readable and Upgoer5 currently exist to report on readability of text, they report the complexity of only a single document. In addition, these tools do not focus on complexity in a more academic-type context.
To address this, we created a tool that uses a data-driven approach to provide authors with insights into the readability of the entirety of their published scholarly work with regard to other text repositories. The tool first quantifies an existing text repository [@Soldatova:2007] with complexity shown to be comparable to that of other scientific journals. The tool subsequently uses this output as a reference to show how the readability of user-selected written work compares to this source.
Ultimately, this tool will expand upon current readability metrics by computing a more detailed and comparative look at the complexity of written text. We hope that this will allow scientists and other experts to better monitor the complexity of their writing relative to other text types, leading to the creation of more accessible online material. And perhaps more broadly contribute to an improved global communication and understanding of complex topics.
Author: [Russell Jarvis](https://github.com/russelljjarvis)\n
Author: [Patrick McGurrin](https://github.com/mcgurrgurr)\n


"""
import nltk

nltk.download("punkt")
nltk.download("cmudict")

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

not_want_list.extend(
    ["link", "librarian", "issue", "abstract", "science", "cookie", "publication"]
)

from science_access.online_app_backend import call_from_front_end
from science_access.online_app_backend import ar_manipulation


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
max = np.max(rd_df["Reading_Level"])
rd_df = rd_df.loc[sample(list(rd_df.index), 999)]
rd_df = rd_df[(rd_df["Reading_Level"] > 0)]

with open("trainingDats.p", "rb") as f:
    trainingDats = pickle.load(f)
    art_df, bio_chem_level, biochem_labels = grab_data_for_splash(trainingDats)
biochem_labels = art_df["Origin"]
bio_chem_level = art_df["Reading_Level"]

# @st.cache(suppress_st_warning=True)
def check_cache(author_name: str, verbose=0):  # ->Union[]
    with shelve.open("fast_graphs_splash.p") as db:
        #flag = author_name in db
        flag = False
        if not flag:
            ar = call_from_front_end(author_name)
            scraped_labels, author_score = frame_to_lists(ar)

            ##
            # This shelve
            # caching wont scale on heroku.
            # need TinyDb on Amazon
            ##
            if len(db.keys()) < 11:
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

        # experimental = [
        #    np.mean([a["standard_len"], a["ndc"]])
        #    for a in ar
        #    if "standard_len" in a.keys()
        # ]
    return ar, author_score, scraped_labels


def show_author_alias(ar: List = []) -> None:
    """
    Synopsis show the hardest to read passage from the entire query to the app user.
    """
    largest = 0
    li = 0
    for i, a in enumerate(ar):
        if "aliases" in a.keys():
            st.markdown(a["aliases"])
            break
    return None


def show_hardest_passage(ar: List = []) -> str:
    """
    Synopsis show the hardest to read passage from the entire query to the app user.
    """
    largest = 0
    li = 0
    smallest = 0
    mean = np.mean([a["standard"] for i, a in enumerate(ar)])

    for i, a in enumerate(ar):
        if a["standard"] >= largest and len(ar[i]["hard_snippet"]):
            largest = a["standard"]
            li = i
        if a["standard"] < smallest:
            smallest = a["standard"]

    for i, a in enumerate(ar):
        if a["standard"] == largest or a["standard"] > mean:

            if "hard_snippet" in ar[i].keys() and ar[i]["hard_snippet"] is not None:
                if len(ar[i]["hard_snippet"]):
                    if (
                        str("can log in with their society credentials")
                        not in ar[i]["hard_snippet"]
                    ):


                        st.markdown("---")

                        st.error("### Some hard to read passage(s) from the authors work.")
                        from nltk import word_tokenize

                        tokens = word_tokenize(ar[i]["hard_snippet"])
                        #string_from_tokens0 = str([str(i)+str(" ") for i in tokens[0:90]])
                        #string_from_tokens2 = "..."
                        #string_from_tokens1 = str([str(i)+str(" ") for i in tokens[-90::]])

                        #string_from_tokens = string_from_tokens0 +string_from_tokens2 + string_from_tokens1
                        string_from_tokens = create_giant_strings(tokens, not_want_list)


                        st.warning(string_from_tokens)  # [0:200])
                        st.warning("...")  # [0:200])

                        return ar[i]
    return None


def clouds_big_words(sci_corpus):
    if len(sci_corpus) != 0:

        st.markdown(""" ### Word Length Word Cloud 	""")
        st.markdown(
            """
		This word cloud is based on the largest words found in the mined text.
		These words are likely culprits that hindered readability.
		"""
        )
        big_words, word_counts_fz, fig_wl = art_cloud_wl(sci_corpus)


verbose = 0

from science_access.t_analysis import text_proc

def main():
    st.title("Search Reading Complexity of an Author")
    st.sidebar.title("Code Information")

    author_name = st.text_input("Enter Author Name:")
    st.markdown(
        """Entering a middle initial may improve search accuracy."""
    )
    st.markdown("-----")

    ar = None
    if author_name:
        ar, author_score, scraped_labels = check_cache(author_name, verbose)
        if len(ar) == 0:
            st.warning("Author Not Found")
    if ar is not None:
        df_author, merged_df = data_frames_from_scrape(
            ar, author_name, scraped_labels, author_score, art_df
        )

    
    my_expander = st.sidebar.beta_expander("click to expand")
    my_expander.markdown(
        """This search applies [dissemin](https://dissemin.readthedocs.io/en/latest/api.html) API backend."""
    )
    
    my_expander.markdown("[Source Code](https://github.com/mcgurrgurr/ScienceAccess) for this tool.")
    my_expander.markdown("[Source Code](https://github.com/russelljjarvis/ScienceAccess) and [link to](https://share.streamlit.io/russelljjarvis/scienceaccess/app.py) an expansion of this tool that is currently in progress.")
    my_expander.markdown("[Rationale for this project.](https://github.com/russelljjarvis/ScienceAccess/blob/master/Documentation/BioRxiv.md)")

    if "df_author" in locals():

        st.markdown("-----")
        #st.markdown("#### Results")

        st.markdown(
            """
			### There were a total number of {0} documents mined during this query.
			""".format(
                len(df_author)
            )
        )
	
        push_frame_to_screen(df_author, scraped_labels)
	
        st.markdown("\n\n")
        st.markdown("\n\n")
	
        if len(author_score) == 0:
            st.warning("Author Not Found")
        
        st.markdown(
            """
			### The average reading level of this work is {0}, with a median level of {2}.""".format(
                round(np.mean(author_score)), 3,round(np.median(author_score)), 3
            )
        )
        
        """
		Reading level is comparable to grade level. 
		For reference, [the average adult reads at an 8th grade reading level](http://nces.ed.gov/naal/pdf/2006470.pdf).
		"""
	
        df_concat_art = pd.concat([rd_df,df_author])
        df_concat_art = pd.concat([df_concat_art,art_df])

        fig_art = px.box(
            df_concat_art, x="Origin", y="Reading_Level", points="all", color="Origin"
        )
        st.write(fig_art)
	
	
        st.markdown(""" The boxplots above show data for the scraped author with two other sources. The first is [ART Corpus]  (https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/), 
		     an existing library of publicly licenced scientific papers. The second is a [collection of papers](https://github.com/elifesciences-publications/readabilityinscience) 
		     analyzed in the paper entitled [The Readability of scientific texts is decreasing over time](https://elifesciences.org/articles/27725#s4). It is an article published in eLife exploring the readability of scientific articles. 
		    """)
		     
		    
        #temp = "{0} Summary Readability versus large sample of science".format(
        #    author_name
        #)
        #labels = [temp, "ART Corpus readability"]
        #values = [np.mean([r["standard"] for r in ar]), np.mean(bio_chem_level)]
        #fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
        #st.write(fig)


        # df_concat_art = pd.concat([art_df, df_author])
        # fig_art = px.box(
        #    df_concat_art, x="Origin", y="Reading_Level", points="all", color="Origin"
        # )
        # st.write(fig_art)

        df0 = df_concat_art


        # try:

        st.markdown("\n")
	
        if np.mean(author_score) < np.mean(bio_chem_level):
            st.markdown(
                """
			{0} was on average easier to read relative to the ART Corpus.
			""".format(
                    author_name
                )
            )

        if np.mean(author_score) >= np.mean(bio_chem_level):
            st.markdown(
                """
			{0} was on average more difficult to read relative to the ART Corpus.
			""".format(
                    author_name
                )
            )
	
        st.markdown("\n")

        if np.mean(author_score) < np.mean(rd_level):
            st.markdown(
                """
			{0} was on average easier to read relative to the collection of papers analyzed in The Readability of Scientific Texts is Decreasing Over Time.
			""".format(
                    author_name
                )
            )

        if np.mean(author_score) >= np.mean(rd_level):
            st.markdown(
                """
			{0} was on average more difficult to read relative to the collection of papers analyzed in The Readability of Scientific Texts is Decreasing Over Time.
			""".format(
                    author_name
                )
            )
	
        my_expander = st.beta_expander("Expand for more information about readability")
        # if my_expander:

        my_expander.markdown(
            """
		### Here are a few additional established text sources of known complexity.
		Note that in general, we can equate reading level with grade level.
		"""
        )

        my_expander.markdown(
            """
		| Text Source | Mean Complexity | Description |
		|----------|----------|:-------------:|
		| [Upgoer 5](https://splasho.com/upgoer5/library.php)                             | 7     | library using only the 10,000 most commonly occurring English words |
		| [Readability of science declining](https://elifesciences.org/articles/27725)   |  9.0 | example of a scientific article discussing writing to a broad audience in an academic context |
		| [Science of writing](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf) | 14.0 | example of a scientific article discussing writing to a broad audience in an academic context |
		| Wikipedia                                                                       | 14.9   | free, popular, crowdsourced encyclopedia generated from self-nominating volunteers  |
		| [Post-Modern Essay Generator](http://www.elsewhere.org/journal/pomo/)           | 16.5   | generates output consisting of sentences that obey the rules of written English, but without restraints on the semantic conceptual references   |
		| [Art Corpus](https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/) | 18.68  | library of scientific papers published in The Royal Society of Chemistry |
		"""
        )

        my_expander.markdown("\n\n")

        my_expander.markdown(
            """
		[Readability Metric Alogrithms and Background](https://en.wikipedia.org/wiki/Readability)
		[Gunning Fog Readability Metric Alogrithm](https://en.wikipedia.org/wiki/Gunning_fog_index)
		#### [Here is a source](http://nces.ed.gov/naal/pdf/2006470.pdf) about variation in adult literacy:
		Kutner M, Greenberg E, Baer J. National Assessment of Adult Literacy (NAAL): A First Look at the Literacy of America’s Adults in the 21st Century (NCES 2006-470). Washington, DC: National Center for Education Statistics; 2005.
		"""
        )
	
	
        st.markdown("\n\n")
        st.markdown("-----")
        st.markdown(""" ### Word Frequency Word Cloud""")
        """
		This word cloud is based on the most common words found in the mined text.
		It is for humans to validate the text mining work. As the word cloud 
		frequency often matches a writer's own knowledge of concepts in their work, 
		it may also help instill trust in text-mining results.
		"""

        grab_setr = []
        grab_set_auth = []

        for paper in trainingDats:
            grab_setr.extend(paper["tokens"])

        for paper in ar:
            grab_set_auth.extend(paper["tokens"])
        artset = list(grab_setr)
        #artset.extend(not_want_list)
        # auth_set = grab_set_auth
        # exclusive = [i for i in grab_set_auth if i not in artset]
        fig = fast_art_cloud(grab_set_auth)

        giant_string = create_giant_strings(grab_set_auth,not_want_list)
        #st.markdown(giant_string)
        urlDat = text_proc(giant_string,urlDat={},verbose=False)

        #st.markdown(urlDat.values())
        #st.markdown(urlDat.keys())

        #st.markdown("### Not Biased By (short) Length of Abstracts Readability Estimate:")
        #st.markdown(urlDat["standard"])
        #hard = show_hardest_passage(ar)
        # if hard is not None:
        # 	st.markdown(hard)
        show_author_alias(ar)

        st.markdown("-----")
        # fast_art_cloud(sci_corpus)
        clouds_by_big_words = True
        if clouds_by_big_words:
            grab_set_auth = []
            for paper in ar:
                if "semantic" in paper.keys():
                    grab_set_auth.extend(paper["tokens"])
            sci_corpus = create_giant_strings(grab_set_auth, not_want_list)
            try:
                clouds_big_words(sci_corpus)
            except:
                pass

        if verbose:
            st.text(sci_corpus)
        with shelve.open("fast_graphs_splash.p") as db:
            if not author_name in db.keys():
                db[author_name] = {
                    "ar": ar,
                    "scraped_labels": scraped_labels,
                    "author_score": author_score,
                    "sci_corpus": sci_corpus,
                }
        st.markdown("\n")
       
        st.markdown("-----")
        st.markdown("\n\n\n\n")

        # sci_corpus = create_giant_strings(ar, not_want_list)
        # bio_corpus = create_giant_strings(trainingDats, not_want_list)

        # st.markdown('Here is one of the biggest words: {0}'''.format(str(big_words[0][0])))
        # st.markdown('Here is one of the biggest words: "{0}", you should feed it into PCA of word2vec'.format(str(big_words[0][0])))

        st.markdown("\n")
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
        # st.markdown(
        #    "### Concepts that differentiate {0} from other science".format(
        #        author_name
        #    )
        # )
        # exclusive = create_giant_strings(ar, exclusive)

        # fig = fast_art_cloud(exclusive)
        # st.markdown("-----")

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

        st.markdown("""### Sentiment""")
        st.markdown(
            """It is {} that the mean sentiment of {}'s writing is more postive relative to that of Readability of the ART Corpus.
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

        st.markdown("\n")
        st.markdown("-----")
        st.markdown(""" The tool we describe consists of a text analysis service and an author search service. These services were created by 
	using or extending many existing Free and Open Source (FOS) tools, including streamlit, requests, WordCloud, TextStat, and The Natural Language Tool Kit. 
	The tool has the capability to retrieve journal hosting links and journal article content (both html and PDF) from application programming interfaces 
	(APIs) and journal hosting websites. Several python libraries helped with querying and gaining access to open science scholarly research documents, 
	and python-requests were used to obtain content from three different APIs, including [dissemin](https://dissemin.readthedocs.io/en/latest/api.html), 
	[semantic-scholar](https://www.semanticscholar.org/), and [unpaywall](https://unpaywall.org/faq).
        """)
	
        st.markdown("\n")
        st.markdown("\n")
        st.markdown("License ScienceAccess is released under the CC BY-NC 2.0. By using ScienceAccess, you are agreeing to its usage terms.")
	
if __name__ == "__main__":

    main()
