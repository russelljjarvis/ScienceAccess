#!/bin/bash
apt-get update
python3 -m pip install -r requirements.txt
python3 -m pip install seaborn
python3 -m pip install bs4
python3 -m pip install natsort dask plotly tabulate
python3 -m conda install -c pyviz holoviews bokeh
conda install -c pyviz holoviews bokeh
python3 -m pip install git+https://github.com/pyviz/holoviews.git


git clone https://github.com/pyviz/holoviews.git
cd holoviews; pip install -e .; cd ..;

wget https://www.dropbox.com/s/3h12l5y2pn49c80/traingDats.p?dl=0
mv traingDats.p?dl=0 data/trainingDats.p



mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"russelljarvis@protonmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
