
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
[![DOI](https://zenodo.org/badge/256452850.svg)](https://zenodo.org/badge/latestdoi/256452850)

[Deployed Application](https://share.streamlit.io/mcgurrgurr/scienceaccess/app.py)
<!--[![ScienceAccess](https://circleci.com/gh/russelljjarvis/ScienceAccess.svg?style=svg)](https://app.circleci.com/pipelines/github/russelljjarvis/ScienceAccess/)-->

<p align="center">
  <img src="app_works.gif" width="700" height="450"/>
</p>


# &TLDR
Large collections of big words and big concepts are hard to read. Scientists share ideas using large collections of words. We machine read scientific documents in an effort to compare writers with each other, and to promote simpler writing.

# Overview
<details>

Non-science writing is often expressed with a less complex and more engaging writing style. This makes it accessible to a broad audience. In contrast, there is research demonstrating that the content of scientific writing is often more complex. This is due, at least in part, to the tendency of scientists to use jargon-heavy language. This can make it inaccessible to many individuals.

This dashboard application allows the user to search for a scientific author. It first performs a web search to gather this author's published works. It then shows the readability score for each published work returned from the web scrape as well as an overall score for that individual. In addition, the application returns the most common and complex words used, sentiment, and a comparison to other text sources with known complexity. [Read more about the application here](https://github.com/mcgurrgurr/ScienceAccess/blob/master/Documentation/paper.md).

More accessible science writing allows knowledge to be more readily transferred into public awareness, as well as between scientists of different fields. Objectively describing the complexity of different writing styles allows us to prescribe how to shift academic science writing into a more accessible niche, where science can better compete with pseudo-science and blogs.

If you're interested in [contributing](https://github.com/mcgurrgurr/ScienceAccess/blob/master/CONTRIBUTING.md) or reviewing the [license](https://github.com/mcgurrgurr/ScienceAccess/blob/master/license.md) for use of this application, refer to our documentation to learn more.

</details>


# Running the application from the terminal

<details>

### Install
```
git clone https://github.com/russelljjarvis/ScienceAccess.git
cd ScienceAccess
sudo $(which python3) setup.py install
```

### Run
```
streamlit run app_dev.py
```

</details>

License
ScienceAccess is released under the CC BY-NC 2.0. By using ScienceAccess, you are agreeing to its usage terms.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
        <td align="center"><a href="https://russelljjarvis.github.io/home/"><img src="https://avatars.githubusercontent.com/u/7786645?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Russell Jarvis</b></sub></a><br /><a href="https://github.com/russelljjarvis/ScienceAccess/commits?author=russelljjarvis" title="Code">💻</a> <a href="https://github.com/russelljjarvis/ScienceAccess/commits?author=russelljjarvis" title="Documentation">📖</a> <a href="#ideas-russelljjarvis" title="Ideas, Planning, & Feedback">🤔</a> <a href="#design-russelljjarvis" title="Design">🎨</a> <a href="#infra-russelljjarvis" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    <td align="center"><a href="http://datamodelsanalytics.com"><img src="https://avatars.githubusercontent.com/u/42288570?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Marc Skov Madsen</b></sub></a><br /><a href="https://github.com/russelljjarvis/ScienceAccess/commits?author=MarcSkovMadsen" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/mcgurrgurr"><img src="https://avatars.githubusercontent.com/u/34006725?v=4?s=100" width="100px;" alt=""/><br /><sub><b>pmcg</b></sub></a><br /><a href="https://github.com/russelljjarvis/ScienceAccess/commits?author=mcgurrgurr" title="Code">💻</a> <a href="https://github.com/russelljjarvis/ScienceAccess/commits?author=mcgurrgurr" title="Documentation">📖</a> <a href="#ideas-mcgurrgurr" title="Ideas, Planning, & Feedback">🤔</a> <a href="#design-mcgurrgurr" title="Design">🎨</a> <a href="#infra-mcgurrgurr" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
  </tr>
</table>


This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
