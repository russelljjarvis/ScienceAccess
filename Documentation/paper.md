-----
title: 'Interactive Exploration in the Readability of Science Authors'

tags:
  - readability
  - science communication
  - science writing
  - Natural Language Processing
  
authors:
  - name: Russell Jarvis
    affiliation: Previous PhD Neuroscience, Arizona State University
  - name: Patrick McGurrin
    affiliation: National Institute of Neurological Disorders and Stroke, National Institutes of Health
  - name: Bradley G Lusk
    affiliation: Science The Earth; Mesa, AZ 85201, USA
  - name: Marc Skov Madsen
    affiliation: Data, Models and Analytics Ninja. PhD, CFA, and Lead Data Scientist Developer at Ørsted. 
  - name: Shivam Bansal
    affiliation: Senior Data Scientist, H2O.ai
  - name: Elise King
    affiliation: Field Ecologist, University of Melbourne
  - name: Rebecca Featherston affiliation: Monash University

date: April  2021
Bibliography: paper.bib

### Summary
Here we present a new text analysis tool capable of reviewing the complexity of a scientist’s published work relative to other online-based text repositories. Rather than providing feedback as to the complexity of a single text as previous tools have done, the tool presented here shows the relative complexity across many texts from the same author, while also comparing the readability of the author’s body of work to a variety of other scientific and lay text types.

The goal of this work is to apply a more data-driven approach that provides established academic authors with statistical insights into their body of published peer reviewed work. By monitoring these readability metrics, scientists may be able to cater their writing to reach broader audiences, contributing to an improved global communication and understanding of complex topics.


### Statement of Need
To ensure that writing is accessible to the general population, authors must consider the length of written text, as well as sentence structure, vocabulary, and other language features [@Kutner:2006]. While popular magazines, newspapers, and other outlets purposefully cater language for a wide audience, there is a tendency for academic writing to use more complex, jargon-heavy language when publishing their work in scientific journals [@Plavén-Sigray:2017], a trend that is becoming more evident over time [@Ball:2017]. 

In the age of growing science communication, this tendency for scientists to use more complex language can carry over when writing in more mainstream media, such as blogs and social media. This can make public-facing material difficult to comprehend, undermining efforts to communicate scientific topics to the general public [@Shulman:2020]. This can contribute to a general misunderstanding of scientific concepts and a disconnect from scientists [@Schulman:2020]. 

To address the increasing difficulty in reading science, we created a data-driven tool to provide authors with insights into the readability of their published open access work with respect to a known collection of science documents called the ART corpus [@Soldatova:2007]. This collection and an additional collection derived from the work “the Readability of Science is Declining over time” has been shown to have complexity that is comparable to that of other scientific journals [@Plavén-Sigray:2017].

The tool we describe consists of a text analysis service and an author search service. These services were created by using or extending many existing Free and Open Source (FOS) tools, including streamlit, requests, WordCloud, TextStat, and The Natural Language Tool Kit (NLTK). The tool has the capability to retrieve journal hosting links and journal article content (both html and PDF) from application programming interfaces (APIs) and journal hosting websites. Several python libraries helped with querying and gaining access to open science scholarly research documents, and python-requests were used to obtain content from three different APIs, including [dissemin](https://dissemin.readthedocs.io/en/latest/api.html), [semantic-scholar](https://www.semanticscholar.org/), and [unpaywall](https://unpaywall.org/faq).

### Search Output

By entering an author’s name into the search box, a system of API calls and web resource requests collects and analyzes text written by that author from the dissemin API. Our tool contrasts the queried authors readability with expected science readability scores because it has pre-processed and quantified the existing corpus of publicly licenced scientific texts contained in ART Corpus (creative commons license), as well as a number of available text repositories and common resources with varying complexity (see Table). 

After querying an author, the readability of the author's work is evaluated by a variety of pre-established metrics. We used  “text_standard,” a readability consensus score that returns the average estimated school grade level required to understand the text. Results from the dissem query include the number of documents on which the readability score was based. Each mined text is presented to the user in a table with an active link to the source text. This allows the user to verify that the results are correct and also provides individual readability scores for each text source.


| Text Source | Mean Complexity | Description |
|----------|----------|:-------------:|
| [Upgoer 5](https://splasho.com/upgoer5/library.php)                             | 7     | library using only the 10,000 most commonly occurring English words |
| [Readability of science declining](https://elifesciences.org/articles/27725)   |  9.0 | example of a scientific article discussing writing to a broad audience in an academic context |
| [Science of writing](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf) | 14.0 | example of a scientific article discussing writing to a broad audience in an academic context |
| Wikipedia                                                                       | 14.9   | free, popular, crowdsourced encyclopedia generated from self-nominating volunteers  |
| [Post-Modern Essay Generator](http://www.elsewhere.org/journal/pomo/)           | 16.5   | generates output consisting of sentences that obey the rules of written English, but without restraints on the semantic conceptual references   |
| [Art Corpus](https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/) | 18.68  | library of scientific papers published in The Royal Society of Chemistry | 

The mined text is compared against the ART Corpus as a side-by-side boxplot of readability scores, as well as data points of each individual readability score. This allows the user to easily review the readability of their work with regard to the ART Corpus. 

Sentiment is also calculated to see overall how positive or negative the text scores are and to cross validate the overall approach. It is expected that science papers should have high objectivity and thus low sentiment polarity. Lastly, a word cloud is generated that shows the most commonly used words in the mined text. 

### Reproducibility
This large hierarchy of software dependencies might act as an impediment to reproducibility. To address this issue we did two things. First, we deployed a [live version of the application](https://share.streamlit.io/mcgurrgurr/scienceaccess/app.py). Second, we created a Docker file and associated Docker container that acts as a self-documenting software environment clone. All code used for this tool can be found on GitHub, which can be run by downloading the github repository and then either building a docker container or running it using the terminal.

 
