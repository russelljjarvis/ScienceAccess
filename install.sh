# https://gist.github.com/mikesmullin/2636776
#
#!/bin/bash
# download and install latest geckodriver for linux or mac.
# required for selenium to drive a firefox browser.
sudo /home/user/anaconda3/bin/pip install -r requirements.txt

sudo apt-get install jq
sudo pip install PyPDF2
sudo pip install pycld2
sudo pip install nltk
sudo pip install selenium
sudo pip install delver
sudo pip install pdfminer
sudo pip install pyvirtualdisplay
sudo pip install textstat
sudo pip install fsspec>=0.3.3
sudo pip install textblob
sudo pip install twython
sudo python3 -c "import nltk; nltk.download('punkt')"
sudo python3 -c "import nltk; nltk.download('stopwords')"
sudo bash gecko_install.sh

git clone https://github.com/ckreibich/scholar.py.git
cd scholar.py
mv scholar.py ..

wget https://www.dropbox.com/s/3h12l5y2pn49c80/traingDats.p?dl=0
wget https://www.dropbox.com/s/crarli3772rf3lj/more_authors_results.p?dl=0
wget https://www.dropbox.com/s/x66zf52himmp5ox/benchmarks.p?dl=0
