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
    brew install -y jq 
    brew install -y wget 
    brew cask install -y firefox
    brew install -y python-lxml
    brew install -y geckodriver
	brew install -y chromium-chromedriver 
    curl -s -L https://www.dropbox.com/s/3h12l5y2pn49c80/traingDats.p
    curl -s -L https://www.dropbox.com/s/crarli3772rf3lj/more_authors_results.p?dl=0
    curl -s -L https://www.dropbox.com/s/x66zf52himmp5ox/benchmarks.p?dl=0

elif [[ $(uname) == "Linux" ]]; then
    sudo apt-get update
    sudo apt-get install -y jq wget firefox
    sudo apt-get install -y python-lxml
    sudo apt-get install -y firefox
    wget https://www.dropbox.com/s/3h12l5y2pn49c80/traingDats.p?dl=0
    wget https://www.dropbox.com/s/crarli3772rf3lj/more_authors_results.p?dl=0
    wget https://www.dropbox.com/s/x66zf52himmp5ox/benchmarks.p?dl=0

else
    echo "can't determine OS"
    exit 1
fi

which -s geckodriver
#if [[ $? != 0 ]] ; then
#    sudo bash install_python3.sh
#fi

which -s firefox
#if [[ $? != 0 ]] ; then
#    sudo bash install_python3.sh
#fi
sudo bash install/gecko_install.sh
which -s firefox
which -s geckodriver
