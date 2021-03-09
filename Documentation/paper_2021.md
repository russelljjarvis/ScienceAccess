-----
title: 'A Tool for Assesing the Readability of Scientific Publications'

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

date: March 2021

### Introduction
To ensure that writing is accessible to the general population, authors must consider the length of written text, as well as sentence structure, vocabulary, and other language features [@Kutner:2006]. While popular magazines, newspapers, and other outlets purposefully cater language for a wide audience, there is a tendency for academic writing to use more complex, jargon-heavy language when publishing their work in scientific journals [@Plavén-Sigray:2017], a trend that is becoming more evident over time [@Ball:2017].
In the age of growing science communication, this tendency for scientists to use more complex language can carry over when writing in more mainstream media, such as blogs and social media [@Yong 2010].
This can make public-facing material difficult to comprehend, undermining efforts to communicate scientific topics to the general public [@Shulman:2020]. Information outflow from scientists that is not accessible can contribute to a general misunderstanding of scientific concepts and a disconnect from scientists. To date over half (57%) of the world's population do not think they know much – if anything – about science. We speculated that this may be due to challenges when attempting to read and understand complex scientific topics. In addition, avoidance of legitimate science research may also be reinforced by more readable content on scientific topics being written by other groups. This content  may potentially be less accurate, or even misleading, depending on its source.
While readability tools, such as Readable and Upgoer5 are capable of reporting on readability of text, they report the complexity of only parts of a single document, which must manually be pasted into an appropriate text field in a webpage. To address this, we created a tool that uses a data-driven approach to provide authors with insights into the readability of their published open access work with respect to a known collection of science documents called the ART corpus [@Soldatova:2007]. This collection has been shown to have complexity that is comparable to that of other scientific journals [@Plavén-Sigray:2017].

This dashboard tool incorporates a  text analysis and author search infrastructure by using or extending many existing Free and Open Source (FOS) tools, including streamlit, requests, WordCloud, TextStat, and The Natural Language Tool Kit (NLTK). Most importantly It also needed the capability to frequently and robustly retrieve appropriate journal hosting links and journal article content (both html and PDF) from both APIs, and journal hosting websites. Several Python libraries helped with querying and gaining access to open science scholarly research documents, and library python-requests were used to obtain content from three different APIs, including dissemin[reference], semantic-scholar[reference], and unpaywall[reference].
This large hierarchy of software dependencies might  act as an impediment to reproducibility. To address this issue we did two things. First, we deployed a live version of the application at [https://agile-reaches-20338.herokuapp.com/]. Second, we created a Docker file and associated Docker container that acts as a self-documenting software environment clone. All code used for this tool can be found on GitHub, which can be run by downloading the github repository and then either building a docker container or running it using the terminal.
Reference Texts used for Analysis
By entering an author’s name into the search box, a system of API calls and web resource requests collects and analyzes text written by that author from the dissemin API. Our dashboard tool is able to contrasts the queried authors readability with expected science readability scores because it has pre-processed and quantified the existing corpus of publicly licenced scientific texts contained in ART Corpus - (creative commons license), as well as a number of available text repositories and common resources with varying complexity (Table 1).

| Text Source | Mean Complexity | Description |
|----------|----------|:-------------:|
| [Upgoer 5](https://splasho.com/upgoer5/library.php)                             | 7     | library using only the 10,000 most commonly occurring English words |
| [Readability of science declining](https://elifesciences.org/articles/27725)   |  9.0 | example of a scientific article discussing writing to a broad audience in an academic context |
| [Science of writing](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf) | 14.0 | example of a scientific article discussing writing to a broad audience in an academic context |
| Wikipedia                                                                       | 14.9   | free, popular, crowdsourced encyclopedia generated from self-nominating volunteers  |
| [Post-Modern Essay Generator](http://www.elsewhere.org/journal/pomo/)           | 16.5   | generates output consisting of sentences that obey the rules of written English, but without restraints on the semantic conceptual references   |
| [Art Corpus](https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/) | 18.68  | library of scientific papers published in The Royal Society of Chemistry | 



### Results
 After querying an author, the readability of the author's work is evaluated by a variety of pre-established metrics. We used  “text_standard,” a readability consensus score that returns the average estimated school grade level required to understand the text. This consensus based score proved to be more robust than any single readability metric. For example Flesch-Kincaid readability score did not always provide robust readability scores.
However, the tool also includes many other metrics. Results from the dissem query include the number of documents on which the readability score was based. Each mined text is presented to the user in a table with an active link to the source text. This allows the user to verify that the results are correct and also provides individual readability scores for each text source.
The mined text is compared against the ART Corpus as a histogram binned by readability score,. The tool provides an overlaying of the distribution of ART Corpus with that of the mined author's text. A second plot above the distribution shows the readability score of each individual queried document so it can be visualized relative to the histogram.
Sentiment is also calculated to see overall how positive or negative the text scores are and to cross validate the overall approach. It is expected that science papers should have high objectivity and thus low sentiment polarity. However, very often the queried author is found to have a significantly positive sentiment.
Lastly, two word clouds are also generated that. These show the most commonly used words and the largest words in the mined text. The collection of big words may help to reconcile unfavorable and high readability scores with the actual written words of the scientific author.
### Discussion
Here, we outline a new tool to analyze the complexity of a scientist’s work relative to other online-based text repositories. Rather than providing feedback as to the complexity of a single text as previous tools have done, the tool presented here showsdemonstrates the relative complexity across many texts from the same author, while. The tool also compareings the readability of the author’s body of work to a variety of other text types.This is useful for getting an overview, or review of multiple scientific outputs, from the same author and for comparing collections of written works between disciplines and science at large.
[End word count]
-------------------------------------------
Code stuff (not to be in the paper per JOSS requirement, but on git)
Downloading the application from Github
```
git clone https://github.com/russelljjarvis/ScienceAccess.git
cd ScienceAccess
```
Building and running the container using Docker
Ensure that you have Docker downloaded on your machine. Scraping is memory and CPU intensive. Default docker resources are not suitable for running the application and thus are manually defined in the code below.
```
docker build -t wcomplexity .
docker run --shm-size=3gb --cpus=2.0 --memory=1g --memory-swap=1g --rm wcomplexity
```
This last command will print two hyperlinks to the terminal. Click on the hyperlink that does not contain the substring local host. This will take you to the application in a browser window.
Running the application from the terminal

Manually download and install the Mozilla geckodriver binary.
```
which python3
```
Place output from above command in
```
sudo [output] setup.py install

streamlit run app.py
```
By either build method, the result is a web interface that prompts the user to enter an author name. After this, a text scraping process occurs.

[Verifying the Application From a Live Hosted version](https://agile-reaches-20338.herokuapp.com/)

[References](https://github.com/russelljjarvis/ScienceAccess/Documentation/paper.bib)
#[1] Kutner, Mark, Elizabeth Greenberg, and Justin Baer. "A First Look at the Literacy of America's Adults in the 21st Century. NCES 2006-470." The National Center for Education Statistics. (2006).

#[2] Plavén-Sigray, Pontus, Granville James Matheson, Björn Christian Schiffler, and William Hedley Thompson. "The readability of scientific texts is decreasing over time." Elife. (2017).

#[3] Ball, Philip. "It's not just you: science papers are getting harder to read." Nature News (2017).

#[4] Shulman, H. C., Dixon, G. N., Bullock, O. M., & Colón Amill, D. (2020). The Effects of Jargon on Processing Fluency, Self-Perceptions, and Scientific Engagement. Journal of Language and Social Psychology (2020).

#[5] Kincaid JP, Fishburne RP Jr, Rogers RL, Chissom BS. "Derivation of new readability formulas (Automated Readability Index, FogCount and Flesch Reading Ease Formula) for Navy enlisted personnel".The Institute for Simulation and Training, (1975): 8–75.

#[6] Soldatova, Larisa, and Maria Liakata. "An ontology methodology and cisp-the proposed core information about scientific papers." JISC Project Report (2007).

#[7] Kuhn, Tobias. "The controlled natural language of randall munroe’s thing explainer." International Workshop on Controlled Natural Language. Springer, Cham, (2016).

#[8] Bulhak, Andrew C. "On the simulation of postmodernism and mental debility using recursive transition networks." Monash University Department of Computer Science (1996).

#[9] Gopen, George D., and Judith A. Swan. "The science of scientific writing." American Scientist 78, no. 6 (1990): 550-558.

#[10] Antonin Delpeuch; a3nm; Ryan Lahfa; translatewiki.net; Xia Li-yao; Steph. No.; Lucas Verney; Evarin; p4bl0; jibe-b; Stefan Beck; nemobis; Virgile Andreani; Martin Monperrus; Kenji Maillard; Jill-Jênn Vie; Golls, "Dissem Software" Zendo, 2019

#[11] Plavén-Sigray, P. and Granville, J. and Schiffler, B. and Thompson, W.,
 The readability of scientific texts is decreasing over time,
 Elife, 2017

