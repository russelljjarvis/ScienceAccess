#!/bin/bash
#alias pip2=$(which pip)
#export pip2=$(which pip)

#PIP=$(which pip)
#sudo PIP install PyPDF2
#export HOME=$HOME
#function pip2=$(which pip)

# https://gist.github.com/mikesmullin/2636776
#
# download and install latest geckodriver for linux or mac.
# required for selenium to drive a firefox browser.
sudo apt-get update
sudo apt-get install jq wget firefox

#install_dir="."
json=$(curl -s https://api.github.com/repos/mozilla/geckodriver/releases/latest)
url=$(echo "$json" | jq -r '.assets[].browser_download_url | select(contains("linux64"))')
curl -s -L "$url" | tar -xz
chmod +x geckodriver
sudo cp geckodriver .
sudo cp geckodriver ./app
export PATH=$PATH:$pwd/geckodriver
echo PATH
#echo "installed geckodriver binary in $install_dir"

#sudo bash gecko_install.sh

#wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
#bash miniconda.sh -b -p $HOME/miniconda
#export PATH="$HOME/miniconda/bin:$PATH"
#sudo apt-get install python-lxml
#hash -r
#conda config --set always_yes yes --set changeps1 no
#conda update -q conda
#conda info -a
#pip install -U pip



sudo python3 -m pip install -r requirements.txt
sudo python3 -m pip install seaborn 
sudo python3 -m pip install bs4
sudo python3 -m pip install natsort dask plotly
sudo python3 -c "import nltk; nltk.download('punkt')"
sudo python3 -c "import nltk; nltk.download('stopwords')"

#git clone https://github.com/ckreibich/scholar.py.git

wget https://www.dropbox.com/s/3h12l5y2pn49c80/traingDats.p?dl=0
wget https://www.dropbox.com/s/crarli3772rf3lj/more_authors_results.p?dl=0
wget https://www.dropbox.com/s/x66zf52himmp5ox/benchmarks.p?dl=0
sudo apt-get install -y firefox
which firefox

mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"rjjarvis@asu.edu\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml


# config
VERSION=38.8.0esr
LANG=en-US
PLATFORM=linux64

# Buildpack URL
ARCHIVE_NAME=firefox
FILE_NAME=${ARCHIVE_NAME}-${VERSION}.tar.bz2
BUILDPACK_FIREFOX_PACKAGE="https://download.mozilla.org/?product=firefox-${VERSION}-SSL&os=${PLATFORM}&lang=${LANG}"

mkdir -p $CACHE_DIR
if ! [ -e $CACHE_DIR/$FILE_NAME ]; then
  echo "-----> Fetching Firefox package from ${BUILDPACK_FIREFOX_PACKAGE}"
  curl $BUILDPACK_FIREFOX_PACKAGE -L -o $CACHE_DIR/$FILE_NAME
fi

"from selenium import webdriver;
from selenium.webdriver.firefox.options import Options;
from selenium.common.exceptions import NoSuchElementException;
options = Options();
options.add_argument('--headless')" >> python -c
