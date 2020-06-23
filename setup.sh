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
sudo apt-get update
sudo apt-get install jq wget firefox

#install_dir="."
json=$(curl -s https://api.github.com/repos/mozilla/geckodriver/releases/latest)
url=$(echo "$json" | jq -r '.assets[].browser_download_url | select(contains("linux64"))')
curl -s -L "$url" | tar -xz
chmod +x geckodriver
sudo cp geckodriver .
sudo cp geckodriver ./app
export PATH=$PATH:$pwd/geckodriver
echo PATH
#echo "installed geckodriver binary in $install_dir"

#sudo bash gecko_install.sh

#wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
#bash miniconda.sh -b -p $HOME/miniconda
#export PATH="$HOME/miniconda/bin:$PATH"
#sudo apt-get install python-lxml
#hash -r
#conda config --set always_yes yes --set changeps1 no
#conda update -q conda
#conda info -a
#pip install -U pip



sudo python3 -m pip install -r requirements.txt
sudo python3 -m pip install seaborn 
sudo python3 -m pip install bs4
sudo python3 -m pip install natsort dask plotly
sudo python3 -c "import nltk; nltk.download('punkt')"
sudo python3 -c "import nltk; nltk.download('stopwords')"

#git clone https://github.com/ckreibich/scholar.py.git

wget https://www.dropbox.com/s/3h12l5y2pn49c80/traingDats.p?dl=0
wget https://www.dropbox.com/s/crarli3772rf3lj/more_authors_results.p?dl=0
wget https://www.dropbox.com/s/x66zf52himmp5ox/benchmarks.p?dl=0
sudo apt-get install -y firefox
which firefox

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


# fail fast
# set -e

# debug
set -x

# parse and derive params
BUILD_DIR=$1
CACHE_DIR=$2
ENV_DIR=$3

LP_DIR=`cd $(dirname $0); cd ..; pwd`
FONTS_DIR=`cd "$LP_DIR/fonts"; pwd`

function error() {
  echo " !     $*" >&2
  exit 1
}

function topic() {
  echo "-----> $*"
}

function indent() {
  c='s/^/       /'
  case $(uname) in
    Darwin) sed -l "$c";;
    *)      sed -u "$c";;
  esac
}

# config
# hardcoded as per stable versions available - upgrade in the next release
VERSION_FIREFOX=67.0.1
VERSION_GECKODRIVER=0.24.0
LANG=en-US
PLATFORM=linux64



# Buildpack URL
ARCHIVE_NAME_FIREFOX=firefox
ARCHIVE_NAME_GECKODRIVER=geckodriver
FILE_NAME_FIREFOX=${ARCHIVE_NAME_FIREFOX}-${VERSION_FIREFOX}.tar.bz2
FILE_NAME_GECKODRIVER=${ARCHIVE_NAME_GECKODRIVER}-v${VERSION_GECKODRIVER}-${PLATFORM}.tar.gz
BUILDPACK_FIREFOX_PACKAGE="https://download.mozilla.org/?product=firefox-${VERSION_FIREFOX}-SSL&os=${PLATFORM}&lang=${LANG}"
BUILDPACK_GECKODRIVER_PACKAGE="https://github.com/mozilla/geckodriver/releases/download/v${VERSION_GECKODRIVER}/${FILE_NAME_GECKODRIVER}"



if [ ! -f $CACHE_DIR/PURGED_CACHE_V1 ]; then
  topic "Purging cache"
  rm -rf $CACHE_DIR/apt
  rm -rf $CACHE_DIR/archives
  rm -rf $CACHE_DIR/lists
  touch $CACHE_DIR/PURGED_CACHE_V1
fi

topic "Installing correct dependencies."

APT_CACHE_DIR="$CACHE_DIR/apt/cache"
APT_STATE_DIR="$CACHE_DIR/apt/state"

mkdir -p "$APT_CACHE_DIR/archives/partial"
mkdir -p "$APT_STATE_DIR/lists/partial"

APT_OPTIONS="-o debug::nolocking=true -o dir::cache=$APT_CACHE_DIR -o dir::state=$APT_STATE_DIR"

topic "Updating apt caches"
apt-get $APT_OPTIONS update | indent

for PACKAGE in $PACKAGES; do
  if [[ $PACKAGE == *deb ]]; then
    PACKAGE_NAME=$(basename $PACKAGE .deb)
    PACKAGE_FILE=$APT_CACHE_DIR/archives/$PACKAGE_NAME.deb

    topic "Fetching $PACKAGE"
    curl -s -L -z $PACKAGE_FILE -o $PACKAGE_FILE $PACKAGE 2>&1 | indent
  else
    topic "Fetching .debs for $PACKAGE"
    apt-get $APT_OPTIONS -y --force-yes -d install --reinstall $PACKAGE | indent
  fi
done

mkdir -p $BUILD_DIR/.apt

for DEB in $(ls -1 $APT_CACHE_DIR/archives/*.deb); do
  topic "Installing $(basename $DEB)"
  dpkg -x $DEB $BUILD_DIR/.apt/
done

# Install Firefox
mkdir -p $CACHE_DIR
if ! [ -e $CACHE_DIR/$FILE_NAME_FIREFOX ]; then
  topic "Fetching Firefox package from ${BUILDPACK_FIREFOX_PACKAGE}"
  curl $BUILDPACK_FIREFOX_PACKAGE -L -o $CACHE_DIR/$FILE_NAME_FIREFOX
fi

topic "Extracting Firefox binaries to ${BUILD_DIR}/vendor/${ARCHIVE_NAME_FIREFOX}"
mkdir -p $CACHE_DIR/$ARCHIVE_NAME_FIREFOX
mkdir -p $BUILD_DIR/vendor
tar jxf $CACHE_DIR/$FILE_NAME_FIREFOX -C $CACHE_DIR
mv $CACHE_DIR/$ARCHIVE_NAME_FIREFOX $BUILD_DIR/vendor/

# Install Geckodriver
if ! [ -e $CACHE_DIR/$FILE_NAME_GECKODRIVER ]; then
  topic "Fetching Geckodriver package from ${BUILDPACK_GECKODRIVER_PACKAGE}"
  curl $BUILDPACK_GECKODRIVER_PACKAGE -L -o $CACHE_DIR/$FILE_NAME_GECKODRIVER
fi

topic "Extracting Geckodriver binaries to ${BUILD_DIR}/vendor/${ARCHIVE_NAME_GECKODRIVER}"
mkdir -p $CACHE_DIR/$ARCHIVE_NAME_GECKODRIVER
mkdir -p $BUILD_DIR/vendor/$ARCHIVE_NAME_GECKODRIVER
tar -xzf $CACHE_DIR/$FILE_NAME_GECKODRIVER -C $CACHE_DIR
chmod +x $CACHE_DIR/$ARCHIVE_NAME_GECKODRIVER
mv $CACHE_DIR/$ARCHIVE_NAME_GECKODRIVER $BUILD_DIR/vendor/$ARCHIVE_NAME_GECKODRIVER/

topic "Setting paths"
mkdir -p $BUILD_DIR/.profile.d
cat <<EOF >$BUILD_DIR/.profile.d/000_apt.sh
export PATH="\$HOME/.apt/usr/bin:\$BUILD_DIR/vendor/firefox:\$BUILD_DIR/vendor/geckodriver:\$PATH"
export LD_LIBRARY_PATH="\$HOME/.apt/usr/lib/x86_64-linux-gnu:\$HOME/.apt/usr/lib/i386-linux-gnu:\$HOME/.apt/usr/lib:\$BUILD_DIR/vendor/firefox:\$BUILD_DIR/vendor/geckodriver:\$LD_LIBRARY_PATH"
export LIBRARY_PATH="\$HOME/.apt/usr/lib/x86_64-linux-gnu:\$HOME/.apt/usr/lib/i386-linux-gnu:\$HOME/.apt/usr/lib:\$BUILD_DIR/vendor/firefox:\$BUILD_DIR/vendor/geckodriver:\$LIBRARY_PATH"
export INCLUDE_PATH="\$HOME/.apt/usr/include:\$HOME/.apt/usr/include/x86_64-linux-gnu:\$INCLUDE_PATH"
export CPATH="\$INCLUDE_PATH"
export CPPPATH="\$INCLUDE_PATH"
export PKG_CONFIG_PATH="\$HOME/.apt/usr/lib/x86_64-linux-gnu/pkgconfig:\$HOME/.apt/usr/lib/i386-linux-gnu/pkgconfig:\$HOME/.apt/usr/lib/pkgconfig:\$PKG_CONFIG_PATH"
export TMPDIR="\$BUILD_DIR/tmp"
EOF

export PATH="$BUILD_DIR/.apt/usr/bin:$PATH:\$BUILD_DIR/vendor/firefox:\$BUILD_DIR/vendor/geckodriver:\$PATH"
export LD_LIBRARY_PATH="$BUILD_DIR/.apt/usr/lib/x86_64-linux-gnu:$BUILD_DIR/.apt/usr/lib/i386-linux-gnu:$BUILD_DIR/.apt/usr/lib:\$BUILD_DIR/vendor/firefox:\$BUILD_DIR/vendor/geckodriver:\$LIBRARY_PATH"
export LIBRARY_PATH="$BUILD_DIR/.apt/usr/lib/x86_64-linux-gnu:$BUILD_DIR/.apt/usr/lib/i386-linux-gnu:$BUILD_DIR/.apt/usr/lib:\$BUILD_DIR/vendor/firefox:\$BUILD_DIR/vendor/geckodriver:\$LIBRARY_PATH"
export INCLUDE_PATH="$BUILD_DIR/.apt/usr/include:$BUILD_DIR/.apt/usr/include/x86_64-linux-gnu:$INCLUDE_PATH"
export CPATH="$INCLUDE_PATH"
export CPPPATH="$INCLUDE_PATH"
export PKG_CONFIG_PATH="$BUILD_DIR/.apt/usr/lib/x86_64-linux-gnu/pkgconfig:$BUILD_DIR/.apt/usr/lib/i386-linux-gnu/pkgconfig:$BUILD_DIR/.apt/usr/lib/pkgconfig:$PKG_CONFIG_PATH"
export TMPDIR="\$BUILD_DIR/tmp"

#give environment to later buildpacks
#export | grep -E -e ' (PATH|LD_LIBRARY_PATH|LIBRARY_PATH|INCLUDE_PATH|CPATH|CPPPATH|PKG_CONFIG_PATH)='  > "$LP_DIR/export"



#echo "Firefox"
#echo "geckodriver"
#python -c "from selenium import webdriver;from selenium.webdriver.firefox.options import Options;from selenium.common.exceptions import NoSuchElementException;options = Options();options.add_argument('--headless');driver = webdriver.Firefox(options=options)"
#exit 0
