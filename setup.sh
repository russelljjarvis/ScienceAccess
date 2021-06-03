#!/bin/bash
# https://gist.github.com/mikesmullin/2636776
#
# download and install latest geckodriver for linux or mac.
# required for selenium to drive a firefox browser.
apt-get update
apt-get install jq wget chromium-chromedriver firefox
python3 -m pip install -r requirements.txt
python3 -m pip install seaborn
python3 -m pip install bs4
python3 -m pip install natsort dask plotly tabulate
python3 -m conda install -c pyviz holoviews bokeh
conda install -c pyviz holoviews bokeh
python3 -m pip install git+https://github.com/pyviz/holoviews.git

# hack package installs:

git clone https://github.com/pyviz/holoviews.git
cd holoviews; pip install -e .; cd ..;

#git clone https://github.com/kermitt2/grobid_client_python
#cd grobid_client_python; pip install -e .; cp grobid_client.py ..;cd ..;
git clone https://github.com/dissemin/dissemin
cd dissemin; pip install -e .;cd ..;


wget https://ftp.mozilla.org/pub/firefox/releases/45.0.2/linux-x86_64/en-GB/firefox-45.0.2.tar.bz2
wget https://www.dropbox.com/s/3h12l5y2pn49c80/traingDats.p?dl=0
mv traingDats.p?dl=0 data/trainingDats.p

wget https://github.com/mozilla/geckodriver/releases/download/v0.24.0/geckodriver-v0.24.0-linux64.tar.gz
tar -xvzf geckodriver*
chmod +x geckodriver

#git clone https://russelljjarvis@github.com/russelljjarvis/CoauthorNetVis.git
#cd CoauthorNetVis; pip install -e .;cd ..;


#git clone https://github.com/russelljjarvis/readabilityinscience
#cd readabilityinscience; pip install -e .;cd ..;



mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"russelljarvis@protonmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
