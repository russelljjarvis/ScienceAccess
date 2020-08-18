

<p align="center">
  <img src="app_works.gif" width="700" height="450"/>
</p>

[![ScienceAccess](https://circleci.com/gh/russelljjarvis/ScienceAccess.svg?style=svg)](https://app.circleci.com/pipelines/github/russelljjarvis/ScienceAccess/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/russelljjarvis/ScienceAccess/master)

# Overview 
Non-science writing is often expressed with a less complex and more engaging writing style. We believe that this makes it more accessible to a broad audience. In contrast, the content of scientific writing is often more complex. This is due to the tendency of scientists to use more complex, jargon-heavy language. This can make it inaccessible to many individuals. 

This dashboard application allows the user to search for a scientific author, sampling a large variety of web documents, and then view the readability of their published work. This includes both a readability score for each of their individual published works returned from the scrape as well as an overall score. In addition, the application returns the most common and complex words used, sentiment, and a comparison to other text repositories with known complexity. [Read more about the application here](https://github.com/russelljjarvis/ScienceAccess/blob/master/Documentation/paper.md).

Multiple stakeholders benefit when science is communicated with lower complexity. Namely, more accessible science writing allows knowledge to be more readily transferred into public awareness. Objectively describing the character of the different writing styles will allow us to prescribe how to shift academic science writing into a more accessible niche, where science can better compete with pseudo-science, and blogs.

If you're interested in [contributing](https://github.com/russelljjarvis/ScienceAccess/blob/master/CONTRIBUTING.md) or reviewing the [license](https://github.com/russelljjarvis/ScienceAccess/blob/master/license.md) for use of this application, refer to our documentation to learn more. 

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
docker run --shm-size=3gb --cpus=2.0 --memory=1g --memory-swap=1g --rm wcomplexity
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
