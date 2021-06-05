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
  - name: Rebecca Featherston
    affiliation: Monash University

date: April  2021
Bibliography: paper.bib

### Summary
Here we present a new text analysis tool capable of reviewing the complexity of a scientist’s published work relative to other online-based text repositories. Rather than providing feedback as to the complexity of a single text as previous tools have done, the tool presented here shows the relative complexity across many texts from the same author, while also comparing the readability of the author’s body of work to a variety of other scientific and lay text types.

The goal of this work is to apply a more data-driven approach that provides established academic authors with statistical insights into their body of published peer reviewed work. By monitoring these readability metrics, scientists may be able to cater their writing to reach broader audiences, contributing to an improved global communication and understanding of complex topics.


### Statement of Need
To ensure that writing is accessible to the general population, authors must consider the length of written text, as well as sentence structure, vocabulary, and other language features [@Kutner:2006]. While popular magazines, newspapers, and other outlets purposefully cater language for a wide audience, there is a tendency for academic writing to use more complex, jargon-heavy language when publishing their work in scientific journals [@Plavén-Sigray:2017], a trend that is becoming more evident over time [@Ball:2017].

In the age of growing science communication, this tendency for scientists to use more complex language can carry over when writing in more mainstream media, such as blogs and social media. This can make public-facing material difficult to comprehend, undermining efforts to communicate scientific topics to the general public [@Shulman:2020]. This can contribute to a general misunderstanding of scientific concepts and a disconnect from scientists [@Schulman:2020].

To address the increasing difficulty in reading science, we created a data-driven tool to provide authors with insights into the readability of their published open access work with respect to a known collection of science documents called the ART corpus [@Soldatova:2007]. This collection and an additional collection derived from the work “the Readability of Science is Declining over time” has been shown to have complexity that is comparable to that of other scientific journals [@Plavén-Sigray:2017].

The tool we describe consists of a text analysis service and an author search service. These services were created by using or extending many existing Free and Open Source (FOS) tools, including streamlit, requests, WordCloud, TextStat, and The Natural Language Tool Kit (NLTK). The tool has the capability to retrieve journal hosting links and journal article content (both html and PDF) from application programming interfaces (APIs) and journal hosting websites. Several python libraries helped with querying and gaining access to open science scholarly research documents, and python-requests were used to obtain content from three different APIs, including [dissemin](https://gitlab.com/dissemin/dissemin), [semantic-scholar](https://www.semanticscholar.org/), and [unpaywall](https://unpaywall.org/faq).

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

### Scientific Application
This work belongs to an emerging scientific discipline that seeks to quantify and understand scientific writing scientifically. Readability metrics have improved in robustness since the publication of the document "the Readability of Science is Declining over Time"[@Plavén-Sigray:2017]. The data set obtained from the Readability of Science is Declininig over Time[@Plavén-Sigray:2017] used a custom implementation of the Flesch reading grade metric, which sometimes yielded negative results, and may have under reported very high readability results. Given the availability of newer more robust readability metrics, it is important to test and calibrate the newer readability metrics against current scientific documents. Text-stats standard obtains a readability metric by averaging over 9 different readability metrics. The presence of frequency word clouds, and large word word-clouds, and hard passages make it possible to sanity check the text-stat metrics as applied to scientific documents. Word clouds act to validate higher readability metrics. Without the word clouds reading grades of >60 might give cause for doubt.

Generally other science text scraping tools might seek to achieve our results by scraping Google Scholar, an approach which almost usually leads to an un-gratifying experience for the app user, who is forced to solve captchas, in exchange for services. Furthermore, the google scholar robots.txt file, prohibits scraping, and obtaining data from google scholar despite Googles stated non consent does not constitute responsible data stewardship.

We present new and reusable methods for mining scientific literature by utilizing many free and newly available APIs dedicated responsible and free dissemination of open access science documents. The tool we described in this document is extensible such that code from various modules could be re-applied to ask more specific questions about readability of science documents.

Because a user can download results from the science accessibility app. The tool makes it possible to compare two different scientific authors in the same field, and to compare their readability metrics. A comparison between two or more authors of the same field could be used to clarify if some fields really are generally harder to understand, or are some authors in the same field generally harder to understand.

### Caveat
No one has an educational grade level corresponding to 80 or 100 years of education. The reading grade level doesn't directly map onto higher values of readability above 16 etc. Therefore the analogy of readability scores to highest level of education attainment has to be taken somewhat loosely.

### Reproducibility
Software dependencies can act as an impediment to reproducibility. To mitigate reproducibility difficulty caused by a constantly changing network of software dependencies we did two things. First, we deployed a [live version of the application](https://share.streamlit.io/russelljjarvis/scienceaccess/app.py). Second, we created a Docker file and associated Docker container that acts as a self-documenting software environment clone. All code used for this tool can be found on GitHub, which can be run by downloading the github repository and then either building a docker container or running it using the terminal.
