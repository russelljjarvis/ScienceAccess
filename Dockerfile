#FROM steveltn/https-portal:1
FROM python:3.7.4





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
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN pip install pyvirtualdisplay
RUN pip install nltk
RUN pip install selenium==3.8.0
RUN pip install --upgrade pip

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

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN conda --version
RUN conda update --yes conda
RUN apt-get install -y python3-dev gcc
RUN conda install --yes gcc_linux-64
RUN apt-get install -y --no-install-recommends g++ protobuf-compiler     
RUN python -m pip install -U pycld3
# Copy local code to the container image.
# --------------- Install python packages using `pip` ---------------
# Installing requirements this way allows you to leverage cache ADD is cache busting.
RUN bash -c 'echo -e "\
    pycld3\n\
	regex\n\
	pdfminer\n\	
	PyPDF2\n\
	nltk\n\
	selenium\n\
	delver\n\
	pdfminer\n\
	pyvirtualdisplay\n\
	textstat\n\
	fsspec>=0.3.3\n\
	textblob\n\
	twython\n\
	streamlit\n\
	wordcloud\n\
	seaborn\n\
	bs4\n\
	natsort\n\
	dask\n\
	plotly\n\
	tabulate\n\
	chart_studio\n\
	tqdm\n\
	crossref-commons\n\
	sklearn\n\
	pipenv\n\
	" > requirements.txt'
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt \
	&& rm -rf requirements.txt
#RUN pip install --upgrade streamlit


RUN python -c "import nltk;nltk.download('punkt')"
RUN python -c "import nltk;nltk.download('stopwords')"
RUN python -c "import nltk;nltk.download('words')"
ENV APP_HOME /app
WORKDIR $APP_HOME

WORKDIR $APP_HOME/data
RUN wget https://www.dropbox.com/s/3h12l5y2pn49c80/traingDats.p?dl=0
RUN wget https://www.dropbox.com/s/x66zf52himmp5ox/benchmarks.p?dl=0
RUN mv traingDats.p?dl=0 traingDats.p
RUN mv benchmarks.p?dl=0 benchmarks.p
# This may be more correct app doesn't mind
# WORKDIR $APP_HOME																	
RUN pip install spacy
RUN python -m spacy download en_core_web_sm \
    && python -m spacy download en_core_web_md \
    && python -m spacy download de_core_news_sm

RUN apt-get update && apt-get install -y procps libsm6 libxext6 libxrender-dev libglib2.0-0 
ADD . .
#ADD requirements.txt ./

#RUN groupadd -g 2000 go 
#RUN useradd -m -u 2001 -g go go
#RUN useradd -ms /bin/bash go
#RUN mkdir -p go_dir
#RUN chown -R go .
#USER go

# --------------- Configure Streamlit ---------------
RUN mkdir -p /root/.streamlit
RUN touch /root/.streamlit/config.toml

#RUN wget https://raw.githubusercontent.com/MarcSkovMadsen/awesome-streamlit/master/.streamlit/config.prod.toml >> /root/.streamlit/config.toml
#RUN wget https://raw.githubusercontent.com/MarcSkovMadsen/awesome-streamlit/master/.streamlit/config.local.toml >> /root/.streamlit/config.toml
RUN touch /root/.streamlit/credentials.toml
RUN echo "[general]" >> /root/.streamlit/credentials.toml
RUN echo 'email = "colouredstatic@gmail.com"' >> /root/.streamlit/credentials.toml

RUN bash -c 'echo -e "\
	[server]\n\
	enableCORS = false\n\
	enableXsrfProtection = false\n\
	\n\
	[browser]\n\
	serverAddress = \"0.0.0.0\"\
	" > /root/.streamlit/config.toml'

EXPOSE 8501


# --------------- Export envirennement variable ---------------
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
#RUN streamlit version
# enviroment variable ensures that the python output is set straight
# to the terminal without buffering it first
ENV PYTHONUNBUFFERED 1
CMD ["streamlit", "run", "--server.port", "8501", "main.py"]
