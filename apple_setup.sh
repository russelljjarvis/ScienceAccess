#!/bin/bash

# https://gist.github.com/mikesmullin/2636776
#
# download and install latest geckodriver for linux or mac.
# required for selenium to drive a firefox browser.

if [[ $(uname) == "Darwin" ]]; then
    ruby -e “$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)” 
    sudo brew install jq wget firefox python-lxml


    
elif [[ $(uname) == "Linux" ]]; then
    sudo apt-get update
    sudo apt-get install jq wget firefox
    sudo apt-get install python-lxml
    sudo apt-get install -y firefox
    which firefox

    #echo "installed geckodriver binary in $install_dir"



else
    echo "can't determine OS"
    exit 1
fi
sudo bash gecko_install.sh




wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
pip install -U pip



sudo python3 -m pip install -r requirements.txt
sudo python3 -m pip install seaborn bs4 natsort dask plotly
sudo python3 -c "import nltk; nltk.download('punkt')"
sudo python3 -c "import nltk; nltk.download('stopwords')"

#git clone https://github.com/ckreibich/scholar.py.git

wget https://www.dropbox.com/s/3h12l5y2pn49c80/traingDats.p?dl=0
wget https://www.dropbox.com/s/crarli3772rf3lj/more_authors_results.p?dl=0
wget https://www.dropbox.com/s/x66zf52himmp5ox/benchmarks.p?dl=0


