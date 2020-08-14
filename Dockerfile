ARG BASE_REPO=ideonate/streamlit-base
FROM $BASE_REPO

RUN pip install streamlit-launchpad>=0.0.6
RUN pip install streamlit --upgrade

EXPOSE 8501
ENV NB_USER jovyan
ENV DISPLAY=:99
ENV DBUS_SESSION_BUS_ADDRESS=/dev/null
USER root
RUN echo "${NB_USER} ALL=NOPASSWD: ALL" >> /etc/sudoers
# FROM python:3.7
# https://github.com/joyzoursky/docker-python-chromedriver/blob/master/py3/py3.6-xvfb-selenium/Dockerfile
RUN apt-get update 
# install selenium
RUN apt-get install -y python3-software-properties
RUN apt-get install -y software-properties-common
RUN apt-get -y install apt-transport-https ca-certificates
RUN apt-get -y install apt-transport-https curl
RUN apt-get -y install wget curl
RUN apt-get install -y firefox-esr
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        bzip2 \
        libfontconfig \
    && apt-get clean

RUN apt-get install --fix-missing
RUN pip install nltk
RUN pip install selenium==3.8.0
RUN pip install --upgrade pip
# set dbus env to avoid hanging
ENV DISPLAY=:99
ENV DBUS_SESSION_BUS_ADDRESS=/dev/null
#RUN pip install --upgrade selenium

##
# Programatic Firefox driver that can bind with selenium/gecko.
##
RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.23.0/geckodriver-v0.23.0-linux64.tar.gz
RUN tar -xvzf geckodriver-v0.23.0-linux64.tar.gz
RUN sh -c 'tar -x geckodriver -zf geckodriver-v0.23.0-linux64.tar.gz -O > /usr/bin/geckodriver'
RUN chmod +x /usr/bin/geckodriver
RUN rm geckodriver-v0.23.0-linux64.tar.gz

RUN cp geckodriver /usr/local/bin/
ENV PATH /usr/bin/geckodriver:$PATH
RUN pip install pyvirtualdisplay
# A lot of academic text is still in PDF, so better get some tools to deal with that.
#RUN sudo /opt/conda/bin/pip install git+https://github.com/pdfminer/pdfminer.six.git


ENV MOZ_HEADLESS = 1
RUN python - c "from selenium import webdriver;\
from selenium.webdriver.firefox.options import Options; \
options = Options(); \
options.headless = True; \
driver = webdriver.Firefox(options=options) ;\
driver.get('http://google.com/') ;\
print('Headless Firefox Initialized') ;\
driver.quit();"

RUN apt-get update
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN conda --version
RUN conda update --yes conda
RUN conda install --yes gcc_linux-64
EXPOSE 8501
ADD . .
ADD requirements.txt ./
#USER jovyan
ADD setup.sh ./
RUN bash setup.sh

#CMD ["streamlit", "--server.port", "8501", "hello"]
#CMD ["streamlit", "run", "--server.port", "8501", "app.py"]
#ENTRYPOINT []
#CMD ["sh", "./setup.sh"]
ENTRYPOINT ["streamlit", "run", "--server.port", "8501", "app.py"]
#CMD ["streamlit", "run", "app.py"]
#ENTRYPOINT ["streamlit", "run","app.py"]
#CMD streamlit run app.py 
# RUN apt-get update
# RUN apt-get install -y python3-dev
# RUN pip install -r requirements.txt
# RUN pip install nltk
# RUN python -c "import streamlit"
# RUN python -c "import bs4"
# RUN python -c "import nltk; nltk.download('punkt');from nltk import word_tokenize,sent_tokenize"
# RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
# RUN python setup.py install; 
# RUN pip install tqdm
# RUN python - c "evidence gecko can work"
# EXPOSE 8080
# EXPOSE 8501
# RUN python -c "import streamlit"
# RUN python -c "import nltk; nltk.download('punkt');from nltk import word_tokenize,sent_tokenize"
# RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
# ADD setup.sh .
# RUN conda install --yes -c conda-forge wordcloud
# RUN conda install --yes -c syllabs_admin pycld2

# RUN conda install --yes pycld2
# CMD ["streamlit", "run", "--server.port", "8501", "app.py"]

