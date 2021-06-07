

<p align="center">
  <img src="app_works.gif" width="700" height="450"/>
</p>

# [Try a live sample of the application](https://share.streamlit.io/russelljjarvis/scienceaccess/app_dev.py)


[![ScienceAccess](https://circleci.com/gh/russelljjarvis/ScienceAccess.svg?style=svg)](https://app.circleci.com/pipelines/github/russelljjarvis/ScienceAccess/)
[![DOI](https://zenodo.org/badge/256452850.svg)](https://zenodo.org/badge/latestdoi/256452850)


# Overview 
Non-science writing is often expressed with a less complex and more engaging writing style. This makes it accessible to a broad audience. In contrast, there is research demonstrating that the content of scientific writing is often more complex. This is due, at least in part, to the tendency of scientists to use jargon-heavy language. This can make it inaccessible to many individuals. 

This dashboard application allows the user to search for a scientific author. It first performs a web search to gather this author's published works. It then shows the readability score for each published work returned from the web scrape as well as an overall score for that individual. In addition, the application returns the most common and complex words used, sentiment, and a comparison to other text sources with known complexity. [Read more about the application here](https://github.com/russelljjarvis/ScienceAccess/blob/master/Documentation/paper.md).

More accessible science writing allows knowledge to be more readily transferred into public awareness, as well as between scientists of different fields. Objectively describing the complexity of different writing styles allows us to prescribe how to shift academic science writing into a more accessible niche, where science can better compete with pseudo-science and blogs.

If you're interested in [contributing](https://github.com/russelljjarvis/ScienceAccess/blob/master/CONTRIBUTING.md) or reviewing the [license](https://github.com/russelljjarvis/ScienceAccess/blob/master/license.md) for use of this application, refer to our documentation to learn more. 

# Running the application from the terminal

# Install
```
git clone https://github.com/russelljjarvis/ScienceAccess.git
cd ScienceAccess
sudo $(which python3) setup.py install
```

```
streamlit run app.py
```


License
ScienceAccess is released under the CC BY-NC 2.0. By using ScienceAccess, you are agreeing to its usage terms.
