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


if [[ $(uname) == "Darwin" ]]; then
    ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" < /dev/null 2> /dev/null
    brew install jq

elif [[ $(uname) == "Linux" ]]; then
    sudo apt-get install jq
else
    echo "can't determine OS"
    exit 1
fi


sudo python3 -m pip install -r requirements.txt
sudo python3 -m pip install PyPDF2
sudo python3 -m pip install pycld2
sudo python3 -m pip install nltk
sudo python3 -m pip install selenium
sudo python3 -m pip install delver
sudo python3 -m pip install pdfminer
sudo python3 -m pip install pyvirtualdisplay
sudo python3 -m pip install textstat
sudo python3 -m pip install fsspec>=0.3.3
sudo python3 -m pip install textblob
sudo python3 -m pip install twython
sudo python3 -c "import nltk; nltk.download('punkt')"
sudo python3 -c "import nltk; nltk.download('stopwords')"
sudo bash gecko_install.sh

#git clone https://github.com/ckreibich/scholar.py.git
#cd scholar.py
#mv scholar.py ..

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