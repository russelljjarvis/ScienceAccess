#!/bin/bash
alias pip2=$(which pip)
export pip2=$(which pip)
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


sudo /home/user/anaconda3/bin/pip install -r requirements.txt
sudo /home/user/anaconda3/bin/pip PyPDF2
sudo /home/user/anaconda3/bin/pip pycld2
sudo /home/user/anaconda3/bin/pip install nltk
sudo /home/user/anaconda3/bin/pip install selenium
sudo /home/user/anaconda3/bin/pip install delver
sudo /home/user/anaconda3/bin/pip install pdfminer
sudo /home/user/anaconda3/bin/pip install pyvirtualdisplay
sudo /home/user/anaconda3/bin/pip install textstat
sudo /home/user/anaconda3/bin/pip install fsspec>=0.3.3
sudo /home/user/anaconda3/bin/pip install textblob
sudo /home/user/anaconda3/bin/pip install twython
sudo python -c "import nltk; nltk.download('punkt')"
sudo python -c "import nltk; nltk.download('stopwords')"
sudo bash gecko_install.sh

#git clone https://github.com/ckreibich/scholar.py.git
#cd scholar.py
#mv scholar.py ..

wget https://www.dropbox.com/s/3h12l5y2pn49c80/traingDats.p?dl=0
wget https://www.dropbox.com/s/crarli3772rf3lj/more_authors_results.p?dl=0
wget https://www.dropbox.com/s/x66zf52himmp5ox/benchmarks.p?dl=0
 