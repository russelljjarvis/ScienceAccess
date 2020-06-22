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
sudo apt-get install jq wget

#install_dir="."
json=$(curl -s https://api.github.com/repos/mozilla/geckodriver/releases/latest)
url=$(echo "$json" | jq -r '.assets[].browser_download_url | select(contains("linux64"))')
curl -s -L "$url" | tar -xz
chmod +x geckodriver
sudo cp geckodriver .
sudo cp geckodriver ./app
export PATH=$PATH:$pwd/geckodriver
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
sudo python3 -m pip install natsort dask
sudo python3 -c "import nltk; nltk.download('punkt')"
sudo python3 -c "import nltk; nltk.download('stopwords')"

#git clone https://github.com/ckreibich/scholar.py.git

wget https://www.dropbox.com/s/3h12l5y2pn49c80/traingDats.p?dl=0
wget https://www.dropbox.com/s/crarli3772rf3lj/more_authors_results.p?dl=0
wget https://www.dropbox.com/s/x66zf52himmp5ox/benchmarks.p?dl=0
 
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