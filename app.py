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

import base64
from nltk import word_tokenize


not_want_list.extend(
    ["link", "librarian", "issue", "abstract", "science", "cookie", "publication"]
)
from plotly.tools import FigureFactory as FF

from science_access.online_app_backend import call_from_front_end
from science_access.online_app_backend import ar_manipulation
from science_access.t_analysis import text_proc
from science_access.online_app_backend import semantic_scholar_alias


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
    extra_options,
)
from science_access.online_app_backend import semantic_scholar_alias

##
# load in readabilityofscience delcining data set.
##
rd_df = pd.read_csv("data/Figure4_SourceData1.csv")

rd_df.rename(
    columns={"flesch_fulltexts": "Reading_Level", "journal": "Origin"}, inplace=True
)

rd_df = rd_df[["Reading_Level", "Origin"]]
def dontcleankeepdirty(rd_df):
    # previously I deleted negative values, but keeping the nonesensical measurements illustrates our point.
    rd_df = rd_df.loc[sample(list(rd_df.index), 999)]
    rd_df = rd_df[(rd_df["Reading_Level"] <80)]

    rd_df = rd_df[(rd_df["Reading_Level"] >= 10)]
    return rd_df
rd_df=dontcleankeepdirty(rd_df)

def cleankeepdirty(rd_df):
    rd_df.rename(
        columns={"flesch_fulltexts": "Reading_Level", "journal": "Origin"}, inplace=True
    )
    # previously I deleted negative values, but keeping the nonesensical measurements illustrates our point.
    #rd_df = rd_df.loc[sample(list(rd_df.index), 999)]
    rd_df = rd_df[(rd_df["Reading_Level"] <100)]
    rd_df = rd_df[(rd_df["Reading_Level"] >10)]

    return rd_df

rd_df["Origin"] = ["ReadabilityScienceDeclining" for i in rd_df["Origin"]]

rd_labels = rd_df["Origin"]
rd_level = rd_df["Reading_Level"]
max = np.max(rd_df["Reading_Level"])



def get_table_download_link_csv(
    object_to_download, author_name, corpus=False, full_text=False
):
    """
    https://discuss.streamlit.io/t/heres-a-download-function-that-works-for-dataframes-and-txt/4052
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    # download_link_text = author_name
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()
    if not corpus:
        if full_text:
            author_name = str("Download full text readability ") + author_name + str(".csv")
        else:
            author_name = str("Download readability ") + author_name + str(".csv")
    else:
        author_name = str("Download collated bag of words file ") + author_name + str(".csv")
    return f'<a href="data:file/txt;base64,{b64}" download="{author_name}">{author_name}</a>'


with open("data/trainingDats.p", "rb") as f:
    trainingDats = pickle.load(f)
    art_df, bio_chem_level, biochem_labels = grab_data_for_splash(trainingDats)
biochem_labels = art_df["Origin"]
bio_chem_level = art_df["Reading_Level"]

def check_cache(author_name: str, verbose=0):
    with shelve.open("data/fast_graphs_splash.p") as db:
        flag = author_name in db
    if not flag:
        try:
            ar = call_from_front_end(author_name, tns=12, fast=True)
            scraped_labels, author_score = frame_to_lists(ar)
        except:
                ar = call_from_front_end(author_name, tns=6, fast=True)
                scraped_labels, author_score = frame_to_lists(ar)
                try:
                    ar = call_from_front_end(author_name, tns=30, fast=False)
                    big_run_done = False
                    scraped_labels, author_score = frame_to_lists(ar)
                except:
                    pass

    else:
        st.success("""
            We have evaluated this query recently, using cached results...
        """)
        with shelve.open("data/fast_graphs_splash.p") as db:
            flag = author_name in db
            temp = db[author_name]
        ar = temp["ar"]
        if "standard_sci" in temp.keys():
            author_score = temp["standard_sci"]
        if "author_score" in temp.keys():
            author_score = temp["author_score"]
        scraped_labels = temp["scraped_labels"]
    return ar, author_score, scraped_labels, author_score,big_run_done


def show_author_alias(ar: List = []) -> None:
    """
    Synopsis show the hardest to read passage from the entire query to the app user.
    """
    largest = 0
    li = 0
    for i, a in enumerate(ar):
        st.text(a.keys())
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
        if "hard_snippet" in ar[i].keys():
            if ar[i]["hard_snippet"] is not None:
                if a["standard"] >= largest and len(ar[i]["hard_snippet"]):
                    largest = a["standard"]
                    li = i
                if a["standard"] < smallest:
                    smallest = a["standard"]

    for i, a in enumerate(ar):
        if a["standard"] == largest or a["standard"] > mean:

            if "hard_snippet" in ar[i].keys():
                # if ar[i]["hard_snippet"] is not None:
                if ar[i]["hard_snippet"] is not None:

                    if len(ar[i]["hard_snippet"]):
                        if (
                            str("can log in with their society credentials")
                            not in ar[i]["hard_snippet"]
                        ):

                            st.markdown("---")

                            st.error(
                                "### Some hard to read passage(s) from the authors work."
                            )

                            tokens = word_tokenize(ar[i]["hard_snippet"])
                            string_from_tokens = create_giant_strings(
                                tokens, not_want_list
                            )

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
		The biggest words presented here are likely detractors from overall readability.
		"""
        )
        big_words, word_counts_fz, fig_wl = art_cloud_wl(sci_corpus)

def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    text = link  # .split('=')[1]
    return f'<a target="_blank" href="{link}">{text}</a>'

verbose = 0


def main():
    st.title("Search Reading Complexity of an Author")

    author_name = st.text_input("Enter Author Name:")
    #st.markdown("-----")

        
    """
    ar = None
    if author_name:
    ar, author_score, scraped_labels,author_score, big_run_done = check_cache(author_name, verbose)
    if len(ar) == 0:
        st.error("Author Not Found")
        st.warning("Try a different spelling of author name")
    #if ar is not None:

    """
    df_author, merged_df = data_frames_from_scrape(
        ar, author_name, scraped_labels, author_score, art_df
    )
    df_author = df_author[(df_author["Reading_Level"] <100)]
    df_author = df_author[(df_author["Reading_Level"] >10)]




    genre = []
    genre.append("scatter plots")
    genre.append("tables")
    genre.append("pie charts")
    genre.append("word clouds")
    # genre.append("hard passages")
    genre.append("ART reference data")
    try:
        info_expander = st.sidebar.expander("Code Information")
    except:
        st.sidebar.expander = st.sidebar.beta_expander
        info_expander = st.sidebar.expander("Code Information")
        #st.expander = st.beta_expander

    info_expander.markdown(
        """This search applies [dissmin](https://dissemin.readthedocs.io/en/latest/api.html) API backend"""
    )

    info_expander.markdown(
        "Source Code: [Github](https://github.com/russelljjarvis/ScienceAccess)"
    )

    info_expander.markdown(
        """[Rationale for this project](https://github.com/russelljjarvis/ScienceAccess/blob/master/Documentation/BioRxiv.md)"""
    )

    html_string="""
    <table>
      <tr>
        <td align="center"><a href="https://russelljjarvis.github.io/home/"><img src="https://avatars.githubusercontent.com/u/7786645?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Russell Jarvis</b></sub></a><br /><a href="https://github.com/russelljjarvis/ScienceAccess/commits?author=russelljjarvis" title="Code">💻</a> <a href="https://github.com/russelljjarvis/ScienceAccess/commits?author=russelljjarvis" title="Documentation">📖</a> <a href="#ideas-russelljjarvis" title="Ideas, Planning, & Feedback">🤔</a> <a href="#design-russelljjarvis" title="Design">🎨</a> <a href="#infra-russelljjarvis" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
        <td align="center"><a href="http://datamodelsanalytics.com"><img src="https://avatars.githubusercontent.com/u/42288570?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Marc Skov Madsen</b></sub></a><br /><a href="https://github.com/russelljjarvis/ScienceAccess/commits?author=MarcSkovMadsen" title="Code">💻</a></td>
      </tr>
      <td align="center"><a href="https://github.com/mcgurrgurr"><img src="https://avatars.githubusercontent.com/u/34006725?v=4?s=100" width="100px;" alt=""/><br /><sub><b>pmcg</b></sub></a><br /><a href="https://github.com/russelljjarvis/ScienceAccess/commits?author=mcgurrgurr" title="Code">💻</a> <a href="https://github.com/russelljjarvis/ScienceAccess/commits?author=mcgurrgurr" title="Documentation">📖</a> <a href="#ideas-mcgurrgurr" title="Ideas, Planning, & Feedback">🤔</a> <a href="#design-mcgurrgurr" title="Design">🎨</a> <a href="#infra-mcgurrgurr" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    </table>
    """
    info_expander.markdown(html_string, unsafe_allow_html=True)

    if "df_author" in locals():

        #st.markdown("-----")
        # st.markdown("#### Results")

        st.markdown(
            """
			There were a total number of {0} documents mined during this abstract driven query, or the scrape involved text reading complexity
            scores tainted by artifacts.
			""".format(
                len(df_author)
            )
        )
        if len(df_author)==0:
            st.markdown(
                """
    			### Hold tight, conducting an analysis of full texts.
    			""")

        if "tables" in genre:
            df_temp = copy.copy(df_author)
            del df_temp["Origin"]
            #df_temp.rename(columns={"Web_Link": "Title"}, inplace=True)
            #push_frame_to_screen(df_temp, author_score)



            df_links = pd.DataFrame()

            df_links["Web_Link"] = df_temp["Web_Link"]
            df_links["Reading_Level"] = df_temp["Reading_Level"]
            df_links.drop_duplicates(subset="Web_Link", inplace=True)
            df_links["Web_Link"] = df_links["Web_Link"].apply(make_clickable)
            df_links = df_links.to_html(escape=False)
            st.write(df_links, unsafe_allow_html=True)

            #st.table(df_temp)  # , scraped_labels)
            get_table_download_link_csv(df_author,author_name)


            st.markdown(
                """Note below, the reference data set in the "the Science of Writing is Declining Over Time, was measured using a custom Flestch algorithm, and it contains (cleaned) negative values and is downward biased.
			The plot also contains a comparison to the ART corpus data set, which was analysed using the newer textstat standard algorithm.
			"""
            )

        if "scatter plots" in genre:
            ##
            # memory intensive don't do:
            # ref_choice = st.radio("switch reference data",("ART Corpus","Readability of Science is Decreasing Over Time"))
            ##
            # ref_choice = "ART Corpus"
            #ref_choice = "Decline"
            #df_author = df_author[(df_author["Reading_Level"] <55)]

            df_concat_art = pd.concat([art_df, df_author])
            df_concat_art = pd.concat([rd_df, df_concat_art])

            fig_art0 = px.box(
                df_concat_art,
                x="Origin",
                y="Reading_Level",
                points="all",
                color="Origin",
            )
            st.write(fig_art0)

        if len(author_score) == 0:
            st.warning("Author Not Found")
        st.markdown(
            """
			### The average reading level of the mined work was {0}.""".format(
                round(np.mean(author_score)), 3
            )
        )

        """
		We can think of reading level in a similar way to grade level.
		For comparison, [the average adult reads at an 8th grade reading level](http://nces.ed.gov/naal/pdf/2006470.pdf).
		"""
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
			 {0} was on average more difficult to read relative to the [ART Corpus](https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/), an existing library of publicly licenced scientific papers.
			""".format(
                    author_name
                )
            )
        try:
            ri_expander = st.expander("Expand for more information about readability")

        except:
            st.expander = st.beta_expander
            ri_expander = st.expander("Expand for more information about readability")

        ri_expander.markdown(
            """
		### Here are a few additional established text sources of known complexity.
		Note that in general, we can equate reading level with grade level.
		"""
        )

        ri_expander.markdown(
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

        ri_expander.markdown("\n\n")

        ri_expander.markdown(
            """
		[Readability Metric Alogrithms and Background](https://en.wikipedia.org/wiki/Readability)
		[Gunning Fog Readability Metric Alogrithm](https://en.wikipedia.org/wiki/Gunning_fog_index)
		#### [Here is a source](http://nces.ed.gov/naal/pdf/2006470.pdf) about variation in adult literacy:
		Kutner M, Greenberg E, Baer J. National Assessment of Adult Literacy (NAAL): A First Look at the Literacy of America’s Adults in the 21st Century (NCES 2006-470). Washington, DC: National Center for Education Statistics; 2005.
		"""
        )

        st.markdown("\n\n")
        #st.markdown("-----")
        st.markdown(""" ### Word Frequency Word Cloud""")
        """
		This word cloud is based on the most common words found in the mined text.
		It is for humans to validate the text mining work. As the word cloud
		frequency often matches a writer's own knowledge of concepts in their work,
		it may also help instill trust in text-mining results.
		"""

        if "word clouds" in genre:
            grab_setr = []
            grab_set_auth = []

            try:
                for paper in ar:
                    grab_set_auth.extend(paper["tokens"])
                artset = list(grab_setr)

                fig = fast_art_cloud(grab_set_auth)

                for paper in trainingDats:
                    grab_setr.extend(paper["tokens"])

                exclusive = [i for i in grab_set_auth if i not in artset]
            except:
                pass

        """
        ### Download Data:
        """
        st.markdown(
            get_table_download_link_csv(df_author, author_name),
            unsafe_allow_html=True,
        )

        st.markdown(
            get_table_download_link_csv(
                pd.DataFrame([{"tokens": grab_set_auth}]),
                author_name,
                corpus=True,
            ),
            unsafe_allow_html=True,
        )


        if "hard passages" in genre:
            hard = show_hardest_passage(ar)
            if hard is not None:
                st.markdown(
                    """Note this text is slightly more disjoint than the original form. NLP pre-processing means that numbers and chemical notation is stripped from the text"""
                )
                st.markdown(hard)

        #st.markdown("-----")

        if "word clouds" in genre:
            grab_set_auth = []
            for paper in ar:
                if "semantic" in paper.keys():
                    grab_set_auth.extend(paper["tokens"])
            sci_corpus = create_giant_strings(grab_set_auth, not_want_list)
            clouds_big_words(sci_corpus)
        #alias_list = semantic_scholar_alias(author_name)
        #st.text(alias_list)
        # my_expander = st.expander("Full Text Score Re calculation")
        # ft = my_expander.radio("Do Full Text",("Yes","No"))
        # if ft=="Yes":
        # if "full text" in genre:
        #if len(alias_list):

        #    st.markdown(
        #        """## Conduct a slower but more rigorous search of the full texts..."""
        #    )

        #    st.markdown(
        #        """The exact search string match in literature search has an import relationship to the results.
    	#	Here are some different aliases this author may have published under:"""
        #    )
            # for al in alias_list:
            #    st.markdown(al)
        #    alias_list.insert(0, "previously selected name")
        #    author_name1 = st.radio("choose name", alias_list)
        #    if author_name == "previously selected name":
        #        author_name = author_name1
        if not big_run_done:
            full_ar_new = call_from_front_end(author_name, tns=40, fast=False)

            scraped_labels_new, author_score = frame_to_lists(full_ar_new)
            df_author_new, merged_df = data_frames_from_scrape(
                full_ar_new, author_name, scraped_labels, author_score, art_df
            )

            if "df_author_new" in locals():
                if "tables" in genre:
                    st.markdown("""### Here you can see""")
                    st.markdown(
                        """how full texts are longer by nature, longer texts are harder to read, the full text items
    				by the same name having higher reading complexity
    				"""
                    )
                    st.markdown("# Full texts:")

                    #push_frame_to_screen(df_author_new, scraped_labels_new)
                    scraped_labels_new.extend(scraped_labels)

                    st.markdown("# Abstracts:")
                    #st.write(df_author)
                    df_author_new = pd.concat([df_author, df_author_new])
                    df_author_new = df_author_new[(df_author_new["Reading_Level"] < 100)]
                    df_author_new = df_author_new[(df_author_new["Reading_Level"] >10)]

                    st.markdown("# Both:")


                    df_links = pd.DataFrame()

                    df_links["Web_Link"] = df_author_new["Web_Link"]
                    df_links["Reading_Level"] = df_author_new["Reading_Level"]
                    df_links.drop_duplicates(subset="Web_Link", inplace=True)
                    df_links["Web_Link"] = df_links["Web_Link"].apply(make_clickable)
                    df_links = df_links.to_html(escape=False)
                    st.write(df_links, unsafe_allow_html=True)


            #st.write(df_author_new)
            #push_frame_to_screen(df_author_new, scraped_labels_new)

            #show_links == "Yes"

            #ttest_expander = st.expander("Show ttest")
            #show_ttest = ttest_expander.radio("ttest?", ("Yes", "No"))
            #if show_ttest:
            twosample_results = scipy.stats.ttest_ind(author_score,bio_chem_level)

            matrix_twosample = [
                ['', 'Test Statistic', 'p-value'],
                ['Sample Data', twosample_results[0], twosample_results[1]]
            ]
            #import plotly.plotly as py
            #import plotly.graph_objs as go
            twosample_table = FF.create_table(matrix_twosample, index=True)
            st.write(twosample_table)
            #py.iplot(twosample_table, filename='twosample-table')


            #data_expander = st.expander("Show Data Download Links")
            #show_links = data_expander.radio("Download Links?", ("Yes", "No"))


            #if show_links == "Yes":
            st.markdown(
                get_table_download_link_csv(
                    df_author_new, author_name, full_text=True
                ),
                unsafe_allow_html=True,
            )

            df_concat_art_new = pd.concat([rd_df, df_author_new])

            if "scatter plots" in genre:
                fig_art = px.box(
                    df_concat_art_new,
                    x="Origin",
                    y="Reading_Level",
                    points="all",
                    color="Origin",
                    labels="Web_Link",
                )
                st.write(fig_art)

        if verbose:
            st.text(sci_corpus)

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



if __name__ == "__main__":

    main()
