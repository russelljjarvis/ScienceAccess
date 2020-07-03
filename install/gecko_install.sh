#!/bin/bash
# download and install latest geckodriver for linux or mac.
# required for selenium to drive a firefox browser.

json=$(curl -s https://api.github.com/repos/mozilla/geckodriver/releases/latest)
if [[ $(uname) == "Darwin" ]]; then
    url=$(echo "$json" | jq -r '.assets[].browser_download_url | select(contains("macos"))')
elif [[ $(uname) == "Linux" ]]; then
    url=$(echo "$json" | jq -r '.assets[].browser_download_url | select(contains("linux64"))')
else
    echo "can't determine OS"
    exit 1
fi

#if [[ -z "${DEPLOY_ENV}" ]]; then
install_dir="../science_access"
curl -s -L "$url" | tar -xz
chmod +x geckodriver
sudo mv geckodriver $pwd$install_dir
#"../science_access"
cat $pwd"../science_access/geckodriver"
export PATH=$PATH:$pwd$install_dir"/geckodriver"

wget https://ftp.mozilla.org/pub/firefox/releases/45.0.2/linux-x86_64/en-GB/firefox-45.0.2.tar.bz2
tar xvf firefox-45.0.2.tar.bz2
sudo mv firefox $pwd$install_dir
cat $pwd"../science_access/firefox"
export PATH=$PATH:$pwd$install_dir"/firefox"

#else
#    install_dir="/usr/local/bin"
#    curl -s -L "$url" | tar -xz
#    chmod +x geckodriver
#    sudo mv geckodriver "$install_dir"
#    echo "installed geckodriver binary in $install_dir"
#fi
