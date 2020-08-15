



[![ScienceAccess](https://circleci.com/gh/russelljjarvis/ScienceAccess.svg?style=svg)](https://app.circleci.com/pipelines/github/russelljjarvis/ScienceAccess/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/russelljjarvis/ScienceAccess/master)

This dashboard application allows the user to search for a scientific author, and to view the readability of their published work. This includes a readability score for each of their published works returned from the scrape, most common and complex words used, sentiment, and a comparison to other known benchmarks of scientific works. 

<img src="app_works.gif" width="450" height="450"/>



### Running the application from the terminal
```
git clone https://github.com/russelljjarvis/ScienceAccess.git
cd ScienceAccess

```
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

### Running the application using Docker
```
git clone https://github.com/russelljjarvis/ScienceAccess.git
cd ScienceAccess

```
# Build the container

```
docker build -t wcomplexity .
```
# Run the container
scraping is memory and CPU defalt docker resources won't cut it:
```
docker run --shm-size=3gb --cpus=4.0 --memory=1g --memory-swap=1g --rm wcomplexity
```


**[Here is a Link to the manuscript with more details on the application.](https://github.com/russelljjarvis/ScienceAccess/blob/master/Documentation/paper.md)** 



### Overview

This project was previously hosted at [https://github.com/russelljjarvis/ScienceAccessibility](https://github.com/russelljjarvis/ScienceAccessibility)

Understanding a big word is hard, so when big ideas are written down with lots of big words, this large pile of big words is also hard to understand. 

We used a computer to quickly visit and read many different websites to see how hard each piece of writing was to understand. People may avoid learning hard ideas, only because too many hard words are encountered in the process. We think we can help by explaining the problem with smaller words, and by creating tools to address the problem.

### Why Are We Doing This?
We want to promote clearer and simpler writing in science, by encorouging scientists in the same field to compete with each other over writing more clearly.

### How Are we Doing This?
We are doing this by creating a data-driven tool that can deliver information about readability to a scientisit about their work or the work of scientists. 

### Machine Estimation of Writing Complexity:
The accessibility of written word can be approximated by a computer program that reads over the text and guesses the mental difficulty, associated with comprehending a written document. The computer program maps reading difficult onto a quantity that is informed by the cognitive load of the writing, and the number of years of schooling needed to decode the language in the document. For convenience, we can refer to the difficulty associated with the text as the 'complexity' of the document. 

### How do some well-known texts do?

First, we sample some extremes in writing style, and then we will tabulate results, so we have some nice reference points to help us to make sense of other results. On the lower and upper limits we have: XKCD: [Pushing the limits of extremely readable science](http://splasho.com/upgoer5/library.php) and for some comparison, we wanted to check some [Machine generated postmodern nonesense](http://www.elsewhere.org/pomo/)

We can thiink of the complexity score similar to reading or grade level.

| Text Source | Mean Complexity | Description |
|----------|----------|:-------------:|
| [Upgoer 5](https://splasho.com/upgoer5/library.php)                             | 7     | library using only the 10,000 most commonly occurring English words |
| [readability of science declining](https://elifesciences.org/articles/27725)   |  9.0 | example of a scientific article discussing writing to a broad audience in an academic context |
| [science of writing](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf) | 14.0 | example of a scientific article discussing writing to a broad audience in an academic context |
| Wikipedia                                                                       | 14.9   | free, popular, crowdsourced encyclopedia generated from self-nominating volunteers  |
| [Post-Modern Essay Generator](http://www.elsewhere.org/journal/pomo/)           | 16.5   | generates output consisting of sentences that obey the rules of written English, but without restraints on the semantic conceptual references   |
| [Art Corpus](https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/) | 18.68  | library of scientific papers published in The Royal Society of Chemistry |


### Developer Overview 
Non-science writing is often expressed with a less complex and more engaging writing style. We believe that this makes it more accessible relative to other academic science writing. This means that reading scientific writing is often more complicated, making it unaccessible to many individuals. 

Unfortunately, writing styles intended for different audiences are predictably different. We show that computers can sample a large variety of web documents, and then document their sentiment, complexity, and other metrics that define its readability.

Multiple stakeholders benefit when science is communicated with lower complexity. Namely, more accessible science writing allows knowledge to be more readily transferred into public awareness. 

Objectively describing the character of the different writing styles will allow us to prescribe how to shift academic science writing into a more accessible niche, where science can more aggressively compete with pseudo-science, and blogs.

