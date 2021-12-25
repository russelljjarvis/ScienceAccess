# SComplexity.t_analysis
# from SComplexity import online_app_backend
# import argparse

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
import os

import streamlit as st
import math
import scipy
from types import MethodType
from nltk import word_tokenize


# from science_access.t_analysis import not_want_list

# word limit smaller than 1000 gets product/merchandise sites.
from science_access.online_app_backend import call_from_front_end
from science_access.online_app_backend import ar_manipulation
from science_access.word_cloud_by_word_len import generate_from_lengths
from science_access.utils import check_passive
import plotly.graph_objects as go

from typing import List, Any
import pandas as pd

theme = px.colors.diverging.Portland
colors = [theme[-1], theme[-2]]

NBINS = 40


def passiveness(acorpus):

    return check_passive(acorpus)


def frame_to_lists(ar):
    scraped_labels = [str(x["link"]) for x in ar]
    standard_sci = [t["standard"] for t in ar]
    return scraped_labels, standard_sci


def try_and_update_cache(ar, trainingDats):
    """
    Try to make the distribution accumulate information based on future queries.
    """
    with open("data/trainingDats.p", "wb") as f:
        st.write(str(type(trainingDats)))
        st.write(str(type(ar)))
        st.write("if types are data frame/list wrangling will be required")
        trainingDats.extend(ar)
        pickle.dump(f, trainingDats)


def get_table_download_link(df):
    """Generates a link allowing the data
    in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'


def zipf_plot(word_counts_fz):
    import streamlit as st

    """
	# https://www.kaggle.com/kaitlyn/zipf-s-law
	f, ax = plt.subplots(figsize=(7, 7))
	ax.set(xscale="log", yscale="log")
	b = sns.regplot("n", "word_rank", word_counts_fz, ax=ax, scatter_kws={"s": 100})
	b.axes.set_title("Zipf Curve")#,fontsize=50)
	b.set_xlabel("logx word frequency")#,fontsize=30)
	b.set_ylabel("logy word frequency")#,fontsize=20)
	b.tick_params(labelsize=5)

	st.pyplot()
	f, ax = plt.subplots(figsize=(7, 7))
	ax.set(xscale="log", yscale="log")
	#st.text(str(word_counts_fz.columns))
	b = sns.scatterplot(word_counts_fz["word_rank"], word_counts_fz["word"], ax=ax)
	b.axes.set_title("Zipf Curve")#,fontsize=50)
	b.set_xlabel("logx word frequency")#,fontsize=30)
	b.set_ylabel("logy word frequency")#,fontsize=20)
	b.tick_params(labelsize=5)
	st.pyplot()
	"""
    f, ax = plt.subplots(figsize=(7, 7))
    ax.set(xscale="log", yscale="log")
    # st.text(str(word_counts_fz.columns))
    b = sns.scatterplot(word_counts_fz["word_rank"], word_counts_fz["n"], ax=ax)
    b.axes.set_title("Zipf Curve")  # ,fontsize=50)
    b.set_xlabel("logx word frequency")  # ,fontsize=30)
    b.set_ylabel("logy word frequency")  # ,fontsize=20)
    b.tick_params(labelsize=5)
    st.pyplot()
    return


# @st.cache
from typing import List


def art_cloud_wl(acorpus):
    WC = WordCloud(background_color="white")
    WC.generate_from_lengths = MethodType(generate_from_lengths, WC)
    fig = plt.figure()
    if type(acorpus) is not type(str()):
        temp_str = ""
        for a in acorpus:
            temp_str += a + " "
        acorpus = temp_str
        # tokens = word_tokenize(acorpus)
        # if len(tokens):
        #    wordcloud = WC.generate_from_lengths(tokens)
    # if type(acorpus) is type(""):
    wordcloud = WC.generate_from_lengths(acorpus)

    if not "wordcloud" in locals():
        return None, None, None
    biggest_words = WC.biggest_words

    word_counts_fz = WC.word_counts_fz
    # wc = WordCloud().generate_from_frequencies(frequencies=di)
    plt.imshow(wordcloud, aspect="auto", interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig, use_column_width=True)
    return biggest_words, word_counts_fz, fig


"""
def zipf_wrapper(acorpus):
    tokens = list(word_tokenize(acorpus))
    zipf_plot(tokens)
"""

# @st.cache


def art_cloud(acorpus):

    # Generate a word cloud image
    WC = WordCloud(background_color="white")

    fig = plt.figure()
    # increase resolution by changing figure size
    # figsize=(25,25))
    # if type(acorpus) is type(list()):
    if type(acorpus) is not type(str()):
        temp_str = ""
        for a in acorpus:
            temp_str += a + " "
        acorpus = temp_str
    wordcloud = WC.generate(acorpus)
    # interpolation "nearest decreases resolution."
    plt.imshow(wordcloud, aspect="auto", interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    return wordcloud, fig, plt


from typing import Any


def fast_art_cloud(acorpus):
    wordcloud, fig, plt = art_cloud(acorpus)
    st.pyplot(fig)
    return fig


def create_giant_strings(ar, not_want_list):
    sci_corpus = ""
    first_pass = []
    if len(ar):
        if type(ar[0]) is type(dict()):
            for t in ar:
                if "tokens" in t.keys():
                    for s in t["tokens"]:
                        if s not in not_want_list:
                            first_pass.append(s)
        else:
            for t in ar:
                if t not in not_want_list:
                    first_pass.append(t)

        first_pass = set(first_pass)
        for s in first_pass:
            if "/" in s:
                temp = s.split("/")  # , " ")
                sci_corpus += str(" ") + temp[0] + str(" ")
                sci_corpus += str(" ") + temp[1] + str(" ")
            if "." in s:
                temp = s.split(".")  # , " ")
                sci_corpus += str(" ") + temp[0] + str(" ")
                sci_corpus += str(" ") + temp[1] + str(" ")
            if s not in set(not_want_list):
                sci_corpus += str(" ") + s + str(" ")  # +str(' ')
    return sci_corpus



def extra_options(ar, trainingDats, df1):

    try:
        # try and update underlying distribution with query, so information about science
        # is culmulative, dynamic.
        try_and_update_cache(ar, trainingDats)
    except:
        st.markdown(str(type(trainingDats)))
        st.markdown(str(type(ar)))
        st.markdown("if types are data frame/list wrangling will be required")
        st.markdown("update cache fails, data wrangling required")
    try:
        # Try to allow researchers of the app to download the data.
        # Via GUI prompts.
        st.markdown(get_table_download_link(df1), unsafe_allow_html=True)
    except:
        st.markdown("try and allow user to download data")


def grab_data_for_splash(trainingDats):

    bio_chem = [t["standard"] for t in trainingDats]
    biochem_labels = [x["file_name"] for x in trainingDats if "file_name" in x.keys()]
    biochem_labels = [x.split("/")[-1] for x in biochem_labels]

    lods = []
    for i, j, k in zip(
        bio_chem,
        [str("Comparison ART Corpus") for i in range(0, len(bio_chem))],
        biochem_labels,
    ):
        lods.append({"Reading_Level": i, "Origin": j, "Web_Link": k})
    df0 = pd.DataFrame(lods)
    return df0, bio_chem, biochem_labels


# @st.cache
def distribution_plot_from_scrape(ar, author_name, scraped_labels, standard_sci, df0):
    # ar = [t for t in ar if t["standard"] < 45]
    group_labels = ["Author: " + str(author_name)]  # , 'Group 2', 'Group 3']
    lods = []
    old_reading_level = 0
    for i, j, k in zip(
        standard_sci, [str(author_name) for i in range(0, len(ar))], scraped_labels
    ):
        if not old_reading_level == i:
            # make sure these are not duplicates with different links
            # exposed as duplicates because their readability scores are identical.
            lods.append({"Reading_Level": i, "Origin": j, "Web_Link": k})
        old_reading_level = i

    df1 = pd.DataFrame(lods)
    df1.drop_duplicates(subset="Web_Link", inplace=True)
    df = pd.concat([df1, df0])
    fig = px.histogram(
        df,
        x="Reading_Level",
        y="Web_Link",
        color="Origin",
        marginal="rug",
        opacity=0.7,  # marginal='violin',# or violin, rug
        hover_data=df.columns,
        hover_name=df["Web_Link"],
        color_discrete_sequence=colors,
        histfunc="count",
        orientation="v",
        nbins=NBINS,
    )

    fig.update_layout(
        title_text="Scraped author {0} versus ART Corpus".format(author_name),
        width=900,
        height=900,
    )
    return df1, fig


def data_frames_from_scrape(ar, author_name, scraped_labels, standard_sci, art_df):
    group_labels = ["Author: " + str(author_name)]
    lods = []

    oc_set = set()
    res = []
    for idx, val in enumerate(standard_sci):
        if val not in oc_set:
            oc_set.add(val)
        else:
            res.append(idx)
    cnt = 0
    for i, j, k in zip(
        standard_sci, [str(author_name) for i in range(0, len(ar))], scraped_labels
    ):
        if cnt not in res:
            # make sure these are not duplicates with different links
            # exposed as duplicates because their readability scores are identical.
            lods.append({"Reading_Level": i, "Origin": j, "Web_Link": k})
        cnt += 1

    df_author = pd.DataFrame(lods)
    df_author.drop_duplicates(subset="Web_Link", inplace=True)
    merged_df = pd.concat([df_author, art_df])
    return df_author, merged_df


# @st.cache
def grand_distribution_plot(ar, scraped_labels, standard_sci, df0, author_name=""):
    # standard_sci = [ t['standard'] for t in ar ]
    # scraped_labels = [ str(x['link']) for x in ar]
    group_labels = ["Author Scraped"]  # , 'Group 2', 'Group 3']
    lods = []
    for i, j, k in zip(
        standard_sci, [str(author_name) for i in range(0, len(ar))], scraped_labels
    ):
        lods.append({"Reading_Level": i, "Origin": j, "Web_Link": k})
    df1 = pd.DataFrame(lods)
    df1.drop_duplicates(subset="Web_Link", inplace=True)
    df = pd.concat([df1, df0])
    x1 = df0["Reading_Level"]
    x2 = df1["Reading_Level"]

    group_labels = ["Comparison Art Corpus ", str(author_name)]

    theme = px.colors.diverging.Portland
    colors = [theme[0], theme[1]]
    colors = [theme[-1], theme[-2]]
    rt = list(pd.Series(scraped_labels))
    fig = ff.create_distplot(
        [x1, x2], group_labels, bin_size=2, colors=colors, rug_text=rt
    )
    hover_trace = [t for t in fig["data"] if "text" in t]
    fig.update_layout(title_text="Scraped author versus ART Corpus")
    fig.update_layout(width=900, height=600)  # , hovermode='x')
    return df1, fig




def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    text = link  # .split('=')[1]
    return f'<a target="_blank" href="{link}">{text}</a>'
    #df_links["Web_Link"] = contents["Web_Link"]
    #df_links["Reading_Level"] = contents["Reading_Level"]
    #df_links.drop_duplicates(subset="Web_Link", inplace=True)
    #df_links["Web_Link"] = df_links["Web_Link"].apply(make_clickable)
    #df_links = df_links.to_html(escape=False)
    #st.write(df_links, unsafe_allow_html=True)


def push_frame_to_screen(contents, readability_vector):# -> pd.DataFrame():
    if type(contents) is type(list()):
        df_links = pd.DataFrame()
        df_links["Web_Link"] = pd.Series(contents)
        df_links["Reading_Level"] = pd.Series(readability_vector)
        df_links.drop_duplicates(subset="Web_Link", inplace=True)
        df_links["Web_Link"] = df_links["Web_Link"].apply(make_clickable)
        df_links = df_links.to_html(escape=False)
        st.write(df_links, unsafe_allow_html=True)
    else:
        df_links = pd.DataFrame()
        #try:
        df_links["Web_Link"] = contents["Web_Link"]
        df_links["Reading_Level"] = contents["Reading_Level"]
        df_links.drop_duplicates(subset="Web_Link", inplace=True)
        df_links["Web_Link"] = df_links["Web_Link"].apply(make_clickable)
        df_links = df_links.to_html(escape=False)
        st.write(df_links, unsafe_allow_html=True)

    return df_links


def get_heights(stats_items, histogram_content, x_sub_set, xys=None):
    vertical_postions_indexs = []
    for i in stats_items:
        vertical_postions_indexs.append(find_nearest(histogram_content, i))
    bin_width_offset = (xys[1][0] - xys[0][0]) / 2.0
    x_sub_set = [i + bin_width_offset for i in x_sub_set]

    heights = []
    for i in vertical_postions_indexs:
        heights.append(xys[i][1])
    return heights, bin_width_offset


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def snap_to_grid(author_stats, bin_centers):
    author_stats_grid = []
    for as_ in author_stats:
        as_ = find_nearest(bin_centers, as_)
        author_stats_grid.append(bin_centers[as_])
    return author_stats_grid


# ipy.stats import norm


def elaborate_plot(trainingDats):
    if not os.path.exists("benchmarks.p?dl=0"):
        os.system(
            "curl -s -L https://www.dropbox.com/s/x66zf52himmp5ox/benchmarks.p?dl=0"
        )
        os.system("wget https://www.dropbox.com/s/x66zf52himmp5ox/benchmarks.p?dl=0")
    with open("benchmarks.p?dl=0", "rb") as f:
        bmark = pickle.load(f)
    try:
        with open("scraped_new.p?dl=0", "rb") as f:
            texts = pickle.load(f)
    except:
        os.system("wget https://www.dropbox.com/s/1kc7alp79h701hx/scraped_new.p?dl=0")
        with open("scraped_new.p?dl=0", "rb") as f:
            texts = pickle.load(f)

    with open("data/_author_specificSayali Phatak.p", "rb") as f:
        ar = pickle.load(f)
        ar = ar[1]

    queries = set([t["query"] for t in texts])
    temp = [t for t in texts if "standard" in t.keys() and "wikipedia" in t["link"]]
    science = [
        "cancer",
        "Vaccines",
        "evolution",
        "climate change",
        "Transgenic",
        "photosysnthesis",
        "evolution",
        "GMO",
    ]
    res = [t["standard"] for t in temp if t["query"] in science]

    mwp = np.mean(res)
    abstract_wiki = {"standard": mwp}
    categories = [
        "Readibility of Science Declining Over Time",
        "Post Modern Essay Generator",
        "upgoer 5",
        "Science of Writing",
        "Mean Wikipedia",
    ]  # "Mean Wikipedia"]#,other_name]# "wikipedia science"]
    max_ART = np.max([t["standard"] for t in trainingDats])
    publication = [t["publication"] for t in trainingDats if t["standard"] == max_ART]
    keys = [t.keys() for t in trainingDats if t["standard"] == max_ART]

    bmark_max_art = {"standard": max_ART}

    standard_sci = [t["standard"] for t in trainingDats]
    print([type(a) for a in ar])

    ar = [t for t in ar if type(t) is type({})]
    ar = [t for t in ar if "standard" in t.keys()]
    xys = [(h.get_x(), h.get_height()) for h in sns.distplot(standard_sci).patches]

    x_grid = [h.get_x() for h in sns.distplot(standard_sci).patches]
    offset = float((x_grid[1] - x_grid[0]) / 2.0)
    bin_centers = [gr + offset for gr in x_grid]
    # this plot not used yet.

    fig = plt.figure(figsize=(10, 8), dpi=80)
    ax1 = fig.add_subplot(111)  # )
    mean_ = np.mean([a["standard"] for a in ar])
    min_ = np.min([a["standard"] for a in ar])
    max_ = np.max([a["standard"] for a in ar])
    std_ = np.std([a["standard"] for a in ar])
    stats_items = [mean_, min_, max_]

    g = sns.distplot(standard_sci, label="Readability Index")

    histogram_content = [x[0] for x in xys]
    height_content = np.array([x[1] for x in xys])
    hc = np.array(histogram_content)
    x_sub_set = histogram_content

    other_name = str(
        "Phytochromobilin C15-Z,syn - C15-E,anti isomerization: concerted or stepwise?"
    )

    standard_sci = [t["standard"] for t in trainingDats]
    ar = [t for t in ar if type(t) is type({})]
    ar = [t for t in ar if "standard" in t.keys()]
    xys = [(h.get_x(), h.get_height()) for h in sns.distplot(standard_sci).patches]

    worst_height, _ = get_heights([max_ART], hc, x_sub_set, xys=xys)
    mwp_height, _ = get_heights([mwp], hc, x_sub_set, xys=xys)

    worst_height = worst_height[0]
    worst_distance = snap_to_grid([max_ART], bin_centers)
    mwp_distance = snap_to_grid([mwp], bin_centers)
    x, y, z = (mwp_distance[0], mwp_height[0], str("mean wikipedia"))

    bmark_stats_items = list(set([b["standard"] for b in bmark]))
    bmark_stats_items.append(x)
    bmark_heights, _ = get_heights(
        bmark_stats_items, histogram_content, x_sub_set, xys=xys
    )
    heights, bwo = get_heights(stats_items, histogram_content, x_sub_set, xys=xys)
    bmark_stats_items = [i + bwo for i in bmark_stats_items]
    mean_a = mean_
    min_a = min_
    max_a = max_
    xticks = list(range(0, 45, 5))

    bmark_stats_items
    box_content = [a["standard"] for a in ar]

    bmark_stats_items_grid = snap_to_grid(bmark_stats_items, bin_centers)
    author_stats = [i for i in [mean_, min_, max_]]
    author_stats_grid = snap_to_grid(author_stats, bin_centers)
    mean_a_grid = snap_to_grid([mean_a], bin_centers)
    x_sub_set_grid = snap_to_grid(x_sub_set, bin_centers)
    recalibrate_heights, b = get_heights(author_stats_grid, hc, x_sub_set, xys=xys)

    heights[0] = np.max(recalibrate_heights)
    heights[2] = recalibrate_heights[2]

    fig, axes = plt.subplots(figsize=(10, 10), nrows=2, ncols=1, sharex=True, dpi=100)
    g = sns.distplot(standard_sci, label="Readability Index")
    if str("data0") not in locals():
        data0 = pd.DataFrame({"mean, min, maximum": author_stats_grid, "CDF": heights})

        data2 = pd.DataFrame(
            {"Standard Reading Level": mean_a_grid, "CDF": np.max(height_content)}
        )
    xinterval1 = copy.copy(bmark_stats_items_grid)
    benchmarks = pd.DataFrame(
        {"benchmarks": bmark_stats_items_grid, "CDF": bmark_heights}
    )

    legend_properties = {"weight": "bold", "size": 8}
    ax = sns.regplot(
        data=benchmarks,
        x="benchmarks",
        y="CDF",
        fit_reg=False,
        marker="o",
        color="green",
    )
    ax = sns.regplot(
        data=data2,
        x="Standard Reading Level",
        y="CDF",
        fit_reg=False,
        marker="o",
        color="red",
    )
    legendMain = ax.legend(
        labels=[str("std deviation")], prop=legend_properties, loc="upper right"
    )
    legendSide0 = ax.legend(labels=[NAME], prop=legend_properties, loc="center right")
    legendSide1 = ax.legend(
        labels=[str("Number of Documents: " + str(len(ar)))],
        prop=legend_properties,
        loc="upper left",
    )
    legendMain = ax.legend(
        labels=[
            str(
                "Google scholar author relative to ART Corpus distribution. Total docs: "
            )
            + str(len(trainingDats))
        ],
        prop=legend_properties,
        loc="upper left",
    )
    #
    x1, y1, z1 = (mwp_distance[0], mwp_height[0], str("mean wikipedia"))

    x, y, z = (worst_distance[0], worst_height, other_name)
    data3 = pd.DataFrame({"Standard Reading Level": [x1], "CDF": [y1]})
    ax = sns.regplot(
        data=data3,
        x="Standard Reading Level",
        y="CDF",
        fit_reg=False,
        marker="o",
        color="green",
    )

    axes[1] = ax = sns.regplot(
        data=benchmarks,
        x="benchmarks",
        y="CDF",
        fit_reg=False,
        marker="o",
        color="green",
    )

    ax2 = plt.twiny()
    xticks = list(range(0, 45, 5))
    ax2.set_xticks(xticks)

    axes[1].set_xticks(xinterval1)
    axes[1].set_xticklabels(categories, minor=False, rotation=90)

    axes[1].axvline(np.mean(standard_sci), color="red", alpha=0.7, linewidth=1.5)
    axes[1].set_ylabel("Probability of Document Reading Level")
    axes[1].set_xlabel("Reading Grade Level")
    bp_dict = axes[0].boxplot(box_content, 0, "gD", vert=False)

    for line in bp_dict["medians"]:
        x, y = line.get_xydata()[1]  # top of median line

    for line in bp_dict["boxes"]:
        x0, y = line.get_xydata()[0]  # bottom of left line
        axes[0].text(
            x0,
            y,
            str(NAME) + " Q1 ",
            horizontalalignment="center",
            verticalalignment="top",
            rotation=90,
        )

        x1, y = line.get_xydata()[3]  # bottom of right line
        axes[0].text(
            x1,
            y,
            str(NAME) + " Q3 ",
            horizontalalignment="center",
            verticalalignment="top",
            rotation=90,
        )

        axes[0].text(
            np.abs(x1 + x0) / 2,
            y,
            str(NAME) + " $\mu$ ",
            horizontalalignment="center",
            verticalalignment="top",
            rotation=90,
        )
        x2, y = line.get_xydata()[1]  # bottom of right line
    axes[0].axvline(np.mean(standard_sci), color="red", alpha=0.7, linewidth=1.5)
    st.pyplot()


"""
TOUR = False
if TOUR:
	NAME1 = args.author1
	online_app_backend.call_from_front_end(NAME,NAME1=author2,tour=TOUR,anon=anon,verbose=verbose)
else:
	NAME = "S S Phatak"
	verbose = False
	online_app_backend.call_from_front_end(NAME,verbose=verbose)
"""
