title: 'A Tool for Assesing the Readability of Scientific Publications'

tags:
  - readability
  - science communication
  - science writing

authors:
  - name: Russell Jarvis
    affiliation: PhD Candidate Neuroscience, Arizona State University
  - name: Patrick McGurrin
    affiliation: National Institute of Neurological Disorders and Stroke, National Institutes of Health
  - name: Shivam Bansal
    affiliation: Senior Data Scientist, H2O.ai
  - name: Bradley G Lusk
    affiliation: Science The Earth; Mesa, AZ 85201, USA
  - name: Elise King
    affiliation: Field Ecologist, University of Melbourne



date: August 2020

bibliography: paper.bib

# Summary
To ensure that writing is accessible to the general population, authors must consider the length of written text, as well as sentence structure, vocabulary, and other language features [@Kutner:2006]. While popular magazines, newspapers, and other outlets purposefully cater language for a wide audience, there is a tendency for academic writing to use more complex, jargon-heavy language [@Plavén-Sigray:2017].

In the age of growing science communication, this tendency for scientists to use more complex language can carry over when writing in more mainstream media, such as blogs and social media. This can make public-facing material difficult to comprehend, undermining efforts to communicate scientific topics to the general public.

While readability tools, such as [Readable](https://www.webfx.com/tools/read-able/) and [Upgoer5](https://splasho.com/upgoer5/) currently exist to report on readability of text, they report the complexity of only a single document. In addition, these tools do not focus on complexity in a more academic-type context.

To address this, we created a tool that uses a data-driven approach to provide authors with insights into the readability of the entirety of their published scholarly work with regard to other text repositories. The tool first quantifies an existing text repository [@Soldatova:2007] with complexity shown to be comparable to that of other scientific journals [Plavén-Sigray:2017]. The tool subsequently uses this output as a reference to show how the readability of user-selected written work compares to this source.

Ultimately, this tool will expand upon current readability metrics by computing a more detailed and comparative look at the complexity of written text. We hope that this will allow scientists and other experts to better monitor the complexity of their writing relative to other text types, leading to the creation of more accessible online material. And perhaps more broadly contribute to an improved global communication and understanding of complex topics.


# Text Analysis Metrics
We built a web-scraping and text analysis infrastructure by extending many existing Free and Open Source (FOS) tools, including Google Scrape, Beautiful Soup, The Natural Language Toolkit, and Selenium. The tool is a dashboard application.

The tool prompts the user to enter an author name, after which a text scraping process occurs.

# Contextualized Readability Output

After performing the web scrape of the entered author's work, the readbaility of the author's work is described using a variety of metrics. The primary metric is the Flesch-Kincaid readability score [@Kincaid:1975]. This readability metric is the most commonly used metric to assess readability. We include the average readability score, and the number of documents that this score was based on. 

The scraped text is also compared relative to Art Corpus [@Soldatova:2007], a library of scientific papers published in The Royal Society of Chemistry. The generated plot for contextualized readability information is a histogram binned by readability score. We use this data because it is a pre-established library of scientific papers. The readability of ART Corpus has also been shown to be comparable to that of other scientific journals [Plavén-Sigray:2017]. The tool provides an overlay of the distribution of Art Corpus with that of the scraped author's text. A secondary distribution plot also shows all of the entered author's works, so that the readability of each individual work can be visualized relative to the histogram.

Each scraped text is presented to the user in a table, so that the user can both verify that the scrape returned corrct results. In addition, the individual readability scores for each of these scraped texts are provided. Each entry in the table is an active link that will also take the user to the source text. 

In addition, we compare the query to a number of available text repositories with varying complexity:

| Text Source | Mean Complexity | Description |
|----------|----------|:-------------:|
| [Upgoer 5](https://splasho.com/upgoer5/library.php)                             | 7     | library using only the 10,000 most commonly occurring English words |
| [Readability of science declining](https://elifesciences.org/articles/27725)   |  9.0 | example of a scientific article discussing writing to a broad audience in an academic context |
| [Science of writing](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf) | 14.0 | example of a scientific article discussing writing to a broad audience in an academic context |
| Wikipedia                                                                       | 14.9   | free, popular, crowdsourced encyclopedia generated from self-nominating volunteers  |
| [Post-Modern Essay Generator](http://www.elsewhere.org/journal/pomo/)           | 16.5   | generates output consisting of sentences that obey the rules of written English, but without restraints on the semantic conceptual references   |
| [Art Corpus](https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/) | 18.68  | library of scientific papers published in The Royal Society of Chemistry |

Upgoer5 [@Kuhn:2016], Wikipedia, and PMEG [@Bulhak:1996] libraries are included to better contextualize the complexity of the ART Corpus data with other text repositories of known complexity. We also include mean readability scores from two scholarly reference papers, Science Declining Over Time [@Kutner:2006] and Science of Writing [@Gopen:1990], which discuss writing to a broad audience in an academic context. We use these to demonstrate the feasibility of discussing complex content using more accessible language, and another benchmark for the user to see when viewing their own readability score.

Sentiment is also calculated to see overall how positive or negative the text scores.  

Lastly, two word clouds are also generated. These show the most commonly used words and the largest words in the scraped text. 


# References
see paper.bib

