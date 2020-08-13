#!/bin/bash
# download and install latest geckodriver for linux or mac.
# required for selenium to drive a firefox browser.

json=$(curl -s https://api.github.com/repos/mozilla/geckodriver/releases/latest)
if [[ $(uname) == "Darwin" ]]; then
    url=$(echo "$json" | jq -r '.assets[].browser_download_url | select(contains("macos"))')
    curl -s -L "$url" |  tar -xz
    wget "$url" 
    tar -xvzf geckodriver*
    chmod +x geckodriver


elif [[ $(uname) == "Linux" ]]; then
    wget https://github.com/mozilla/geckodriver/releases/download/v0.24.0/geckodriver-v0.24.0-linux64.tar.gz
    tar -xvzf geckodriver*  
    chmod +x geckodriver
    #url=$(echo "$json" | jq -r '.assets[].browser_download_url | select(contains("linux64"))')
else
    echo "can't determine OS"
    exit 1
fi


export PATH=$PATH:$pwd"/geckodriver"

wget https://ftp.mozilla.org/pub/firefox/releases/45.0.2/linux-x86_64/en-GB/firefox-45.0.2.tar.bz2
tar xvf firefox-45.0.2.tar.bz2
chmod +x firefox/firefox
chmod +x firefox
#sudo mv firefox $pwd$install_dir
#cat $pwd"../science_access/firefox/firefox"
export PATH=$PATH:$pwd"/firefox/firefox"
#source /etc/environment
#else
#    install_dir="/usr/local/bin"
#    curl -s -L "$url" | tar -xz
#    chmod +x geckodriver
#    sudo mv geckodriver "$install_dir"
#    echo "installed geckodriver binary in $install_dir"
#fi
