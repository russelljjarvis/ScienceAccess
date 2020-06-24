#!/bin/bash
# https://gist.github.com/mikesmullin/2636776
if [[ $(uname) == "Darwin" ]]; then
    ruby -e “$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)” 
    brew install jq 
    brew install wget 
    brew install firefox 
    brew install python-lxml
	brew install chromium-chromedriver 
elif [[ $(uname) == "Linux" ]]; then
    sudo apt-get update
    sudo apt-get install jq wget firefox
    sudo apt-get install python-lxml
    sudo apt-get install -y firefox
else
    echo "can't determine OS"
    exit 1
fi
sudo bash gecko_install.sh
sudo python3 -m pip install -r requirements.txt
sudo python3 -m pip install seaborn bs4 natsort dask plotly
sudo python3 -c "import nltk; nltk.download('punkt')"
sudo python3 -c "import nltk; nltk.download('stopwords')"
wget https://www.dropbox.com/s/3h12l5y2pn49c80/traingDats.p?dl=0
wget https://www.dropbox.com/s/crarli3772rf3lj/more_authors_results.p?dl=0
wget https://www.dropbox.com/s/x66zf52himmp5ox/benchmarks.p?dl=0

# sudo conda install -c plotly plotly=4.8.1
# sudo python3 align_data_sources.py