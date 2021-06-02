
-----
Title: Interactive Exploration in the Readability of Science Authors

tags:
readability
science communication
science writing
Natural Language Processing

authors:
- Russell Jarvis affiliation:Previous PhD Computational Neuroscience, Arizona State University 
- Patrick McGurrin affiliation: National Institute of Neurological Disorders and Stroke, National Institutes of Health
- Bradley G Lusk affiliation: Science The Earth; Mesa, AZ 85201, USA
- Rebecca Featherston affiliation: Research Fellow Department of Social Work Monash University
- Marc Skov Madsen affiliation: Data, Models and Analytics Ninja. PhD, CFA, and Lead Data Scientist Developer at Ørsted.
- Shivam Bansal affiliation: Senior Data Scientist, H2O.ai

date: April 2021

### Introduction
To ensure that writing is accessible to the general population, authors must consider the length of written text, as well as sentence structure, vocabulary, and other language features [@Kutner:2006]. While popular magazines, newspapers, and other outlets purposefully cater language for a wide audience, there is a tendency for academic writing to use more complex, jargon-heavy language when publishing their work in scientific journals [@Plavén-Sigray:2017], a trend that is becoming more evident over time [@Ball:2017]. 

In the age of growing science communication, this tendency for scientists to use more complex language can carry over when writing in more mainstream media, such as blogs and social media [@Yong 2010]. This can make public-facing material difficult to comprehend, undermining efforts to communicate scientific topics to the general public [@Shulman:2020]. Information outflow from scientists that is not accessible can contribute to a general misunderstanding of scientific concepts and a disconnect from scientists [@Schulman:2020]. To date over half (57%) of the world's population do not think they know much – if anything – about science. We speculated that this may be due to challenges when attempting to read and understand complex scientific topics.
To address the increasing difficulty in reading science, we created a data-driven tool to provide authors with insights into the readability of their published open access work with respect to a known collection of science documents called the ART corpus [@Soldatova:2007].  An additional collection derived from the work “the Readability of Science is Declining over time”. has been shown to have complexity that is comparable to that of other scientific journals [@Plavén-Sigray:2017].

The tool we describe consists of a text analysis service and an author search service. These services were created by using or extending many existing Free and Open Source (FOS) tools, including streamlit, requests, WordCloud, TextStat, and The Natural Language Tool Kit (NLTK). The tool has the capability to retrieve journal hosting links and journal article content (both html and PDF) from APIs, and journal hosting websites. Several Python libraries helped with querying and gaining access to open science scholarly research documents, the library python-requests were used to obtain content from three different APIs, including [dissemin](https://gitlab.com/dissemin/dissemin), [semantic-scholar](https://www.semanticscholar.org/), and [unpaywall](https://unpaywall.org/faq).

This large hierarchy of software dependencies might act as an impediment to reproducibility. To address this issue we did two things. First, we deployed a [live version of the application](https://agile-reaches-20338.herokuapp.com/). Second, we created a Docker file and associated Docker container that acts as a self-documenting software environment clone. All code used for this tool can be found on GitHub, which can be run by downloading the github repository and then either building a docker container or running it using the terminal.

### Reference Texts used for Analysis
By entering an author’s name into the search box, a system of API calls and web resource requests collects and analyzes text written by that author from the dissemin API. Our tool  contrasts the queried authors readability with expected science readability scores because it has pre-processed and quantified the existing corpus of publicly licenced scientific texts contained in ART Corpus - (creative commons license), as well as a number of available text repositories and common resources with varying complexity (Table 1). 


| Text Source | Mean Complexity | Description |
|----------|----------|:-------------:|
| [Upgoer 5](https://splasho.com/upgoer5/library.php)                             | 7     | library using only the 10,000 most commonly occurring English words |
| [Readability of science declining](https://elifesciences.org/articles/27725)   |  9.0 | example of a scientific article discussing writing to a broad audience in an academic context |
| [Science of writing](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf) | 14.0 | example of a scientific article discussing writing to a broad audience in an academic context |
| Wikipedia                                                                       | 14.9   | free, popular, crowdsourced encyclopedia generated from self-nominating volunteers  |
| [Post-Modern Essay Generator](http://www.elsewhere.org/journal/pomo/)           | 16.5   | generates output consisting of sentences that obey the rules of written English, but without restraints on the semantic conceptual references   |
| [Art Corpus](https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/) | 18.68  | library of scientific papers published in The Royal Society of Chemistry | 


### Running the Application
[Verifying the Application From a Live Host](https://share.streamlit.io/russelljjarvis/scienceaccess/app.py)
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

### Results

After querying an author, the readability of the author's work is evaluated by a variety of pre-established metrics. We used  “text_standard,” a readability consensus score that returns the average estimated school grade level required to understand the text. Results from the dissem query include the number of documents on which the readability score was based. Each mined text is presented to the user in a table with an active link to the source text. This allows the user to verify that the results are correct and also provides individual readability scores for each text source. 

The mined text is compared against the ART Corpus as a side-by-side boxplot of readability scores, as well as data points of each individual readability score. This allows the user to easily review the readability of their work with regard to the ART Corpus.

Sentiment is also calculated to see overall how positive or negative the text scores are and to cross validate the overall approach. It is expected that science papers should have high objectivity and thus low sentiment polarity. Lastly, a word cloud shows the most commonly used words in the mined text. 

### Conclusions
Here, we outline a new tool to analyze the complexity of a scientist’s work relative to other online-based text repositories. Rather than providing feedback as to the complexity of a single text as previous tools have done, the tool presented here shows the relative complexity across many texts from the same author, while also comparing the readability of the author’s body of work to a variety of other text types.

Our work expands upon these current readability tools and metrics by better contextualizing the complexity of a written document’s text. Rather than providing feedback as to the complexity of a single text as these other tools do, the tool discussed here demonstrates the relative complexity across many texts from the same author , as well as provides comparison to a variety of other text types.

The goal of this work, in contrast to previous work, is to apply a more data-driven approach to provide established academic authors with statistical insights into their body of published peer reviewed work. This will expand upon current readability metrics by computing a more detailed and comparative look at the complexity of written text which allows scientists and other experts to use a data-driven approach to monitor the complexity of their writing relative to other text types, leading to the creation of more accessible online material. By monitoring these readability metrics, scientists may be able to cater their writing to reach broader audiences, contributing to improved global communication and understanding of complex topics.

This tool can apply more broadly to other fields, including philosophy, politics, law, and art. The focus of this work on scientific texts is driven by a desire to  increase public accessibility to scientific journal articles. However, some modifications might be required to fit these other fields (e.g. a web scrape from a different source). We encourage others to use the linked source code to hypothesize, explore, and implement changes to improve the tool and cater it for other fields. 

### Limitations and Future Work
In the context of this work, there are two important caveats. Firstly, science writing consists of two types of complexity: irreducible complexity, and reducible complexity. Irreducible complexity in science writing stems from the fact that science is the act of communicating abstract and elaborate techniques that interrogate nature. Written instructions for how to repeat a science experiment should not be less complex than the experiments that the instructions encode. Reducible complexity in science writing is the not strictly necessary complexity in science writing, that comes from the introduction of difficult language (long jargon heavy sentences many modifiers and written in the passive voice). This extra fraction of complex writing is added to texts by the science writers by accident. In this document we are only concerned with methods that help writers to identify reducible complexity in their writing. 

It is very rare for a single scientist to be a sole author of a publication since teams of scientists often coauthor a publication. The data mining and NLP approach here involves querying  an author’s written work and analysing the co-authored documents that are discovered in this process. Therefore, the writing complexity discovered by this tool reflects the written complexity of the group of scientists in the whole co-authorship network.  This tool assumes that the written contribution of the author can be approximated by averaging over multiple written documents, although we acknowledge that this approximation may decrease the accuracy of the reported readability score for an individual author.

By sharing the project source code we hope that other scientists or engineers will help to expand and improve this tool. Improvements include filtering of mined texts and increasing the content that returns from the text analysis, such as a more detailed breakdown of the returned scientific text. For example, it should be possible to provide highlighted snippets of the most difficult to read passages for a given author.
One other consideration is the idea of calculating the readability score for groups of co-authors. Since scientific documents are often co-authored, the output metrics from the scrapper tool reflect the contributions from a network of co-authors rather a single individual. Capturing the writing contributions of the author searched for and their closest co-authors may help to understand readability with a broader scope.

### References
[1] Kutner, Mark, Elizabeth Greenberg, and Justin Baer. "A First Look at the Literacy of America's Adults in the 21st Century. NCES 2006-470." The National Center for Education Statistics. (2006).

[2] Plavén-Sigray, Pontus, Granville James Matheson, Björn Christian Schiffler, and William Hedley Thompson. "The readability of scientific texts is decreasing over time." Elife. (2017).

[3] Ball, Philip. "It's not just you: science papers are getting harder to read." Nature News (2017).

[4] Shulman, H. C., Dixon, G. N., Bullock, O. M., & Colón Amill, D. (2020). The Effects of Jargon on Processing Fluency, Self-Perceptions, and Scientific Engagement. Journal of Language and Social Psychology (2020).

[5] Kincaid JP, Fishburne RP Jr, Rogers RL, Chissom BS. "Derivation of new readability formulas (Automated Readability Index, FogCount and Flesch Reading Ease Formula) for Navy enlisted personnel".The Institute for Simulation and Training, (1975): 8–75.

[6] Soldatova, Larisa, and Maria Liakata. "An ontology methodology and cisp-the proposed core information about scientific papers." JISC Project Report (2007).

[7] Kuhn, Tobias. "The controlled natural language of randall munroe’s thing explainer." International Workshop on Controlled Natural Language. Springer, Cham, (2016).

[8] Bulhak, Andrew C. "On the simulation of postmodernism and mental debility using recursive transition networks." Monash University Department of Computer Science (1996).

[9] Gopen, George D., and Judith A. Swan. "The science of scientific writing." American Scientist 78, no. 6 (1990): 550-558.
 
[10] Antonin Delpeuch; a3nm; Ryan Lahfa; translatewiki.net; Xia Li-yao; Steph. No.; Lucas Verney; Evarin; p4bl0; jibe-b; Stefan Beck; nemobis; Virgile Andreani; Martin Monperrus; Kenji Maillard; Jill-Jênn Vie; Golls, "Dissem Software" Zendo, 2019

