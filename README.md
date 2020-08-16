



[![ScienceAccess](https://circleci.com/gh/russelljjarvis/ScienceAccess.svg?style=svg)](https://app.circleci.com/pipelines/github/russelljjarvis/ScienceAccess/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/russelljjarvis/ScienceAccess/master)

This dashboard application allows the user to search for a scientific author, and to view the readability of their published work. This includes both a readability score for each of their individual published works returned from the scrape as well as an overall score. In addition, the application returns the most common and complex words used, sentiment, and a comparison to other text repositories with known complexity. 

<img src="app_works.gif" width="450" height="450"/>



# Running the application using Docker
```
git clone https://github.com/russelljjarvis/ScienceAccess.git
cd ScienceAccess

```
### Build the container

```
docker build -t wcomplexity .
```
### Run the container
Scraping is memory and CPU intensive. Default docker resources aren't suitable for running the application.
```
docker run --shm-size=3gb --cpus=4.0 --memory=1g --memory-swap=1g --rm wcomplexity
```
This last command will print two hyperlinks to the terminal. Click on the hyperlink that does not contain the substring local host. This will take you to the application in a browser window.



# Running the application from the terminal
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

**[Here is a Link to the manuscript with more details on the application.](https://github.com/russelljjarvis/ScienceAccess/blob/master/Documentation/paper.md)** 



# Overview
Understanding a big word is hard, so when big ideas are written down with lots of big words, this large pile of big words is also hard to understand. People may avoid learning hard ideas, only because too many hard words are encountered in the process. We think we can help by explaining the problem with smaller words, and by creating tools to address the problem.

### Why Are We Doing This?
We want to promote writing in science that is easy to read. And we feel that there is not a resource that gives scientists adequate feedback on the readbaility of their work. Such a tool may encorouge scientists to write in a way that is more readable to a broad audience.

### How Are we Doing This?
We are doing this by creating a data-driven tool that can deliver information about readability to a scientist about their work or the work of other scientists in their field. Specifically, we used a computer to quickly visit and read many different websites to see how hard each piece of writing was to understand. The computer program maps the content with regard to the number of years of schooling needed to decode the language in the document. Thus, we can think of the complexity score similar to reading or grade level.

### How do some well-known texts do?
After calculating the score for a given author, we display this score with regard to other sampled texts. This gives some reference points to help make sense of the results. For example, on the lower readability side we have Upgoer 5, which [pushes the limits of extremely readable science](http://splasho.com/upgoer5/library.php). On the higher end of readability we include some [machine generated postmodern nonesense](http://www.elsewhere.org/pomo/).


| Text Source | Mean Complexity | Description |
|----------|----------|:-------------:|
| [Upgoer 5](https://splasho.com/upgoer5/library.php)                             | 7     | library using only the 10,000 most commonly occurring English words |
| [readability of science declining](https://elifesciences.org/articles/27725)   |  9.0 | example of a scientific article discussing writing to a broad audience in an academic context |
| [science of writing](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf) | 14.0 | example of a scientific article discussing writing to a broad audience in an academic context |
| Wikipedia                                                                       | 14.9   | free, popular, crowdsourced encyclopedia generated from self-nominating volunteers  |
| [Post-Modern Essay Generator](http://www.elsewhere.org/journal/pomo/)           | 16.5   | generates output consisting of sentences that obey the rules of written English, but without restraints on the semantic conceptual references   |
| [Art Corpus](https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/) | 18.68  | library of scientific papers published in The Royal Society of Chemistry |


# Developer Overview 
Non-science writing is often expressed with a less complex and more engaging writing style. We believe that this makes it more accessible to a broad audience. In contrast, scientific writing is often more complicated. This is due to the tendency of scientists to use more complex, jargon-heavy language. This can make it inaccessible to many individuals. Here we sample a large variety of web documents, and then document their sentiment, complexity, and other metrics that define its readability.

Multiple stakeholders benefit when science is communicated with lower complexity. Namely, more accessible science writing allows knowledge to be more readily transferred into public awareness. Objectively describing the character of the different writing styles will allow us to prescribe how to shift academic science writing into a more accessible niche, where science can more aggressively compete with pseudo-science, and blogs.

