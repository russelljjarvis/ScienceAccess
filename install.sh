# https://gist.github.com/mikesmullin/2636776
# 
#!/bin/bash
# download and install latest geckodriver for linux or mac.
# required for selenium to drive a firefox browser.

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

install_dir="/usr/local/bin"
json=$(wget https://api.github.com/repos/mozilla/geckodriver/releases/latest)
if [[ $(uname) == "Darwin" ]]; then
    url=$(echo "$json" | jq -r '.assets[].browser_download_url | select(contains("macos"))')
    #curl 
elif [[ $(uname) == "Linux" ]]; then
    url=$(echo "$json" | jq -r '.assets[].browser_download_url | select(contains("linux64"))')

else
    echo "can't determine OS"
    exit 1
fi
wget "$url" | tar -xz

chmod +x geckodriver
sudo mv geckodriver "$install_dir"
echo "installed geckodriver binary in $install_dir"


