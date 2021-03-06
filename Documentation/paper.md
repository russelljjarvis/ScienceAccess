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

# Introduction
To ensure that writing is accessible to the general population, authors must consider the length of written text, as well as sentence structure, vocabulary, and other language features [@Kutner:2006]. While popular magazines, newspapers, and other outlets purposefully cater language for a wide audience, there is a tendency for academic writing to use more complex, jargon-heavy language [@Plavén-Sigray:2017]. In fact, this trend is becoming more evident over time [@Ball:2017]. Science writing using accessible language can be challenging to begin with, and is often not a part of formalized academic training. In addition, communicating scientific information is difficult given the overall breadth and niche detail of content.

In the age of growing science communication, this tendency for scientists to use more complex language can carry over when writing in more mainstream media, such as blogs and social media. This can make public-facing material difficult to comprehend, undermining efforts to communicate scientific topics to the general public [@Shulman:2020].

While readability tools, such as [Readable](https://www.webfx.com/tools/read-able/) and [Upgoer5](https://splasho.com/upgoer5/) currently exist to report on readability of text, they report the complexity of only a single document. In addition, these tools do not focus on complexity in a more academic-type context. To address this, we created a tool that uses a data-driven approach to provide authors with insights into the readability of the entirety of their published scholarly work with regard to other text repositories. The tool first quantifies an existing text repository [@Soldatova:2007] with complexity shown to be comparable to that of other scientific journals [@Plavén-Sigray:2017]. The tool subsequently uses this output as a reference to show how the readability of user-selected written work compares to this source.

# Methods

### Text Analysis Metrics
We built a web-scraping and text analysis infrastructure by extending many existing Free and Open Source (FOS) tools, including Google Scrape, Beautiful Soup, The Natural Language Toolkit, and Selenium. The tool is a dashboard application.

### Reference Texts used for Analysis
The results of the web scrape are compared to a number of available text repositories with varying complexity (see table). These include: 

Upgoer5 is a library of scientific texts using only the 10,000 most commonly occurring English words. This restraint, results in output text that that is common among the general public.

Wikipedia is a free, crowdsourced  encyclopedia that is generated from self-nominating volunteers. We include it given it’s popular use for scientific and non-scientific information by the general public.

Postmodern Essay Generator (PMEG) generates output consisting of sentences that obey the rules of written English, but have no restraints on the semantic conceptual references. 

ART Corpus is a pre-existing body of scientific papers published in The Royal Society of Chemistry (RSC) and released with a creative commons license. 

Upgoer5 [@Kuhn:2016], Wikipedia, and PMEG [@Bulhak:1996] libraries are included to better contextualize the complexity of the ART Corpus data with other text repositories of known complexity. We also include mean readability scores from two scholarly reference papers, Science Declining Over Time [@Kutner:2006] and Science of Writing [@Gopen:1990], which discuss writing to a broad audience in an academic context. We use these to demonstrate the feasibility of discussing complex content using more accessible language, and another benchmark for the user to see when viewing their own readability score.


| Text Source | Mean Complexity | Description |
|----------|----------|:-------------:|
| [Upgoer 5](https://splasho.com/upgoer5/library.php)                             | 7     | library using only the 10,000 most commonly occurring English words |
| [Readability of science declining](https://elifesciences.org/articles/27725)   |  9.0 | example of a scientific article discussing writing to a broad audience in an academic context |
| [Science of writing](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf) | 14.0 | example of a scientific article discussing writing to a broad audience in an academic context |
| Wikipedia                                                                       | 14.9   | free, popular, crowdsourced encyclopedia generated from self-nominating volunteers  |
| [Post-Modern Essay Generator](http://www.elsewhere.org/journal/pomo/)           | 16.5   | generates output consisting of sentences that obey the rules of written English, but without restraints on the semantic conceptual references   |
| [Art Corpus](https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/) | 18.68  | library of scientific papers published in The Royal Society of Chemistry | 

### Reproducibility
We note that this large hierarchy of software dependencies can act as a significant impediment to reproducibility. To address this issue, we used a Docker file and associated Docker container together, as they act as a self-documenting and extremely portable software environment clone.

All code used for the creation of the code can be found on [GitHub](https://github.com/russelljjarvis/ScienceAccess/). It can be run by downloading the github repository to your machine, and then be either building a docker container or by running it using the terminal. 

### Downloading the application from Github

```
git clone https://github.com/russelljjarvis/ScienceAccess.git
cd ScienceAccess

```
### Building and running the container using Docker
Ensure that you have Docker downloaded on your machine. Scraping is memory and CPU intensive. Default docker resources aren't suitable for running the application and thus are manually defined in the code below.

```
docker build -t wcomplexity .
docker run --shm-size=3gb --cpus=2.0 --memory=1g --memory-swap=1g --rm wcomplexity
```
This last command will print two hyperlinks to the terminal. Click on the hyperlink that does not contain the substring local host. This will take you to the application in a browser window.

### Running the application from the terminal

```
which python3
```
place output from above command in
```
sudo [output] setup.py install
```

```
streamlit run app.py
```

By either build method, the result is a web interface that prompts the user to enter an author name. After this, a text scraping process occurs.


# Results
After performing the web scrape of the entered author's work, the readbaility of the author's work is described using a variety of metrics. The primary metric is the Flesch-Kincaid readability score [@Kincaid:1975]. This readability metric is the most commonly used metric to assess readability. We include the average readability score, and the number of documents that this score was based on. 

The scraped text is also compared relative to Art Corpus [@Soldatova:2007], a library of scientific papers published in The Royal Society of Chemistry. The generated plot for contextualized readability information is a histogram binned by readability score. We use this data because it is a pre-established library of scientific papers. The readability of ART Corpus has also been shown to be comparable to that of other scientific journals [Plavén-Sigray:2017]. The tool provides an overlay of the distribution of Art Corpus with that of the scraped author's text. A second plot above the distribution shows the readability score of each individual work can be visualized relative to the histogram.

Sentiment is also calculated to see overall how positive or negative the text scores.  

Lastly, two word clouds are also generated. These show the most commonly used words and the largest words in the scraped text.

Each scraped text is presented to the user in a table, so that the user can both verify that the scrape returned correct results. In addition, the individual readability scores for each of these scraped texts are provided. Each entry in the table is an active link that will also take the user to the source text.

# Conclusion
Here we outline a new tool to analyze the complexity of a scientist’s work relative to other online-based text repositories. By entering an author’s name into the tool, an automated web-scraping process collects and analyzes text written by that author from Google Scholar. 

Other already existing tools, like the Upgoer5 and Hemmingway, use imputed text to generally raise awareness to the general deficiency of readability in writing. Upgoer5 exposes readers to fun playful text as they apply low reading levels to highly technical subject matter. Hemmingway takes this further by notifying the user of suggestions to simplify input text in real time. Our work expands upon these current readability tools and metrics by better contextualzing the complexity of a written document’s text. Rather than only providing feedback as to the complexity of a single text as these other tools do, ours demonstrates the relative complexity across many texts from the same author, as well as provides comparison to a variety of other text types. 

We do not intend to compete with these well-established tools. By contrast, the goal of this work is to apply a more data-driven approach to provide established academic authors with statistical insights into their body of published science work. We believe this will expand upon current readability metrics by computing a more detailed and comparative look at the complexity of written text. We hope that this will allow scientists and other experts to use a data-driven approach to better monitor the complexity of their writing relative to other text types, leading to the creation of more accessible online material. And perhaps more broadly contribute to an improved global communication and understanding of complex topics.

# Future Work
We feel that while the application is accessible using the code mentioned in the methods section, it can be burdensome for a user who is less familiar with coding language and processed. Future work will aim to implement the application as a web tool that can be accessed in a more accessible manner. By linking to the code we also hope that other scientists or engineers will help to expand and improve this tool moving forward. 

# References
see paper.bib

#[1] Kutner, Mark, Elizabeth Greenberg, and Justin Baer. "A First Look at the Literacy of America's Adults in the 21st Century. NCES 2006-470." The National Center for Education Statistics. (2006).

#[2] Plavén-Sigray, Pontus, Granville James Matheson, Björn Christian Schiffler, and William Hedley Thompson. "The readability of scientific texts is decreasing over time." Elife. (2017).

#[3] Ball, Philip. "It's not just you: science papers are getting harder to read." Nature News (2017).

#[4] Shulman, H. C., Dixon, G. N., Bullock, O. M., & Colón Amill, D. (2020). The Effects of Jargon on Processing Fluency, Self-Perceptions, and Scientific Engagement. Journal of Language and Social Psychology (2020).

#[5] Kincaid JP, Fishburne RP Jr, Rogers RL, Chissom BS. "Derivationof new readability formulas (Automated Readability Index, FogCount and Flesch Reading Ease Formula) for Navy enlistedpersonnel".The Institue for Simulation and Training, (1975): 8–75.

#[6] Soldatova, Larisa, and Maria Liakata. "An ontology methodology and cisp-the proposed core information about scientific papers." JISC Project Report (2007).

#[7] Kuhn, Tobias. "The controlled natural language of randall munroe’s thing explainer." International Workshop on Controlled Natural Language. Springer, Cham, (2016).

#[8] Bulhak, Andrew C. "On the simulation of postmodernism and mental debility using recursive transition networks." Monash University Department of Computer Science (1996).  

#[9] Gopen, George D., and Judith A. Swan. "The science of scientific writing." American Scientist 78, no. 6 (1990): 550-558.

#[10] Antonin Delpeuch; a3nm; Ryan Lahfa; translatewiki.net; Xia Li-yao; Steph. No.; Lucas Verney; Evarin; p4bl0; jibe-b; Stefan Beck; nemobis; Virgile Andreani; Martin Monperrus; Kenji Maillard; Jill-Jênn Vie; Golls, "[Dissem Software](https://dissemin.readthedocs.io/en/latest/index.html)" Zendo, 2019
#[11] Plavén-Sigray, P. and Granville, J. and Schiffler, B. and Thompson, W.,
 The readability of scientific texts is decreasing over time,
 Elife, 2017

