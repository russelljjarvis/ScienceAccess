FROM jupyter/scipy-notebook
ENV NB_USER jovyan
ENV DISPLAY=:99
ENV DBUS_SESSION_BUS_ADDRESS=/dev/null
USER root
RUN echo "${NB_USER} ALL=NOPASSWD: ALL" >> /etc/sudoers

# FROM python:3.7
# https://github.com/joyzoursky/docker-python-chromedriver/blob/master/py3/py3.6-xvfb-selenium/Dockerfile
RUN apt-get update && apt-get install -y gnupg
RUN apt-get install -yqq xvfb
# set display port and dbus env to avoid hanging

# install selenium
RUN apt-get update
RUN apt-get install -y python3-software-properties
RUN apt-get install -y software-properties-common
RUN apt-get -y install apt-transport-https ca-certificates
RUN apt-get -y install apt-transport-https curl
RUN apt-get -y install wget curl
RUN apt-get update
RUN apt-get install -y firefox
RUN apt-get update
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        bzip2 \
        libfontconfig \
    && apt-get clean

RUN apt-get update
RUN apt-get install --fix-missing


USER jovyan

RUN pip install nltk
RUN pip install selenium==3.8.0
RUN pip install --upgrade pip
# Upgrade to version 2.0
#RUN conda install -y matplotlib
# Make sure every Python file belongs to jovyan
# Remove dangling symlinks
RUN find -L /opt/conda -type l -delete
# Make sure every Python file is writable

#RUN chown -R $NB_USER $HOME
#RUN rm -rf /var/lib/apt/lists/*
# set dbus env to avoid hanging
ENV DISPLAY=:99
ENV DBUS_SESSION_BUS_ADDRESS=/dev/null
#RUN pip install --upgrade selenium

##
# Programatic Firefox driver that can bind with selenium/gecko.
##
RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.23.0/geckodriver-v0.23.0-linux64.tar.gz
RUN sudo tar -xvzf geckodriver-v0.23.0-linux64.tar.gz
RUN sudo chown -R jovyan $HOME
RUN sudo sh -c 'tar -x geckodriver -zf geckodriver-v0.23.0-linux64.tar.gz -O > /usr/bin/geckodriver'
RUN sudo chmod +x /usr/bin/geckodriver
RUN rm geckodriver-v0.23.0-linux64.tar.gz

RUN sudo chown -R jovyan /home/jovyan
RUN sudo cp geckodriver /usr/local/bin/
ENV PATH /usr/bin/geckodriver:$PATH
#RUN pip install selenium
RUN pip install pyvirtualdisplay
# A lot of academic text is still in PDF, so better get some tools to deal with that.
#RUN sudo /opt/conda/bin/pip install git+https://github.com/pdfminer/pdfminer.six.git

# https://github.com/GoogleScraper.git
#RUN sudo /opt/conda/bin/pip install git+https://github.com/NikolaiT/GoogleScraper
WORKDIR $HOME
#RUN python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
RUN sudo chown -R jovyan $HOME
RUN python -c "import bs4"

ENV MOZ_HEADLESS = 1
RUN python - c "from selenium import webdriver;\
from selenium.webdriver.firefox.options import Options; \
options = Options(); \
options.headless = True; \
driver = webdriver.Firefox(options=options) ;\
driver.get('http://google.com/') ;\
print('Headless Firefox Initialized') ;\
driver.quit();"

RUN sudo chown -R jovyan .
# copying all analysis code to image
COPY . .
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN pip install nltk
#RUN python -c "import streamlit"
RUN python -c "import nltk; nltk.download('punkt');from nltk import word_tokenize,sent_tokenize"
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger')"

RUN python setup.py install; 
RUN sudo chown -R jovyan unit_test
RUN sudo chown -R jovyan .
RUN pip install tqdm

RUN python - c "from selenium import webdriver;\
from selenium.webdriver.firefox.options import Options; \
options = Options(); \
options.headless = True; \
driver = webdriver.Firefox(options=options) ;\
driver.get('http://google.com/') ;\
print('Headless Firefox Initialized') ;\
driver.quit();"
RUN python - c "evidence gecko can work"
#RUN python unit_test/scrape_test.py
WORKDIR $HOME
#RUN sudo mkdir .streamlit
#ADD .streamlit/credentials.toml .streamlit/credentials.toml
RUN sudo chown -R jovyan .  
RUN find /opt/conda ! -writable -print0 | xargs -0 -I {} sudo chmod 744 {}
RUN find /opt/conda ! -user $NB_USER -print0 | xargs -0 -I {} sudo chown -h $NB_USER {}

EXPOSE 8080
#   RUN cat app.py
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
#RUN cat /home/jovyan/.streamlit
USER $NBUSER
EXPOSE 8080
WORKDIR $HOME
USER jovyan
EXPOSE 8501
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
WORKDIR $HOME
CMD ["streamlit", "run", "--server.port", "8501", "app.py"]
RUN python -c "import streamlit"
RUN python -c "import nltk; nltk.download('punkt');from nltk import word_tokenize,sent_tokenize"
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
RUN python3 -c "import streamlit"
RUN python3 -c "import nltk; nltk.download('punkt');from nltk import word_tokenize,sent_tokenize"
RUN python3 -c "import nltk; nltk.download('averaged_perceptron_tagger')"

