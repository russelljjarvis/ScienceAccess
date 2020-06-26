#!/bin/bash
# https://gist.github.com/mikesmullin/2636776
if [[ $(uname) == "Darwin" ]]; then
    which -s brew
    if [[ $? != 0 ]] ; then
        # Install Homebrew
        ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    else
        brew update
    fi
    brew install jq 
    brew install wget 
    brew cask install firefox
    brew install python-lxml
	# brew install chromium-chromedriver 
    curl -s -L https://www.dropbox.com/s/3h12l5y2pn49c80/traingDats.p
    curl -s -L https://www.dropbox.com/s/crarli3772rf3lj/more_authors_results.p?dl=0
    curl -s -L https://www.dropbox.com/s/x66zf52himmp5ox/benchmarks.p?dl=0

elif [[ $(uname) == "Linux" ]]; then
    sudo apt-get update
    sudo apt-get install jq wget firefox
    sudo apt-get install python-lxml
    sudo apt-get install -y firefox
    wget https://www.dropbox.com/s/3h12l5y2pn49c80/traingDats.p?dl=0
    wget https://www.dropbox.com/s/crarli3772rf3lj/more_authors_results.p?dl=0
    wget https://www.dropbox.com/s/x66zf52himmp5ox/benchmarks.p?dl=0

else
    echo "can't determine OS"
    exit 1
fi

which -s pip3
if [[ $? != 0 ]] ; then
    sudo bash install_python3.sh
fi

which -s python3
if [[ $? != 0 ]] ; then
    sudo bash install_python3.sh
fi
sudo bash install/gecko_install.sh
sudo bash install/part2.sh
sudo python install/align_data_sources.py
sudo python -c "import nltk; nltk.download('punkt')"
sudo python -c "import nltk; nltk.download('stopwords')"

