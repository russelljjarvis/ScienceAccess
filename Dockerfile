FROM russelljarvis/science_accessibility_user
USER root   
# streamlit-specific commands
RUN mkdir -p /root/.streamlit
USER jovyan
RUN sudo bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN sudo bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

# exposing default port for streamlit
EXPOSE 8501

# copying all analysis code to image
COPY . .


COPY requirements.txt ./

RUN sudo /opt/conda/bin/pip install -r requirements.txt
RUN sudo /opt/conda/bin/pip install tqdm
RUN sudo /opt/conda/bin/python setup.py install; 
RUN sudo chown -R jovyan unit_test
RUN sudo chown -R jovyan .

RUN python unit_test/scrape_test.py

RUN sudo bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN sudo bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

# activate conda environment
SHELL ["/bin/bash", "-c", "source activate ./env"]

ENTRYPOINT ["streamlit","run","app.py"]
# run app
CMD ["streamlit","run","app.py"]