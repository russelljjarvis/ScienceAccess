
mkdir -p ~/.streamlit/
touch ~/.streamlit/credentials.toml
echo "[general]" >> ~/.streamlit/credentials.toml
echo 'email = "replace_me@gmail.com"' >> ~/.streamlit/credentials.toml
echo "[server]" >> ~/.streamlit/config.toml
#echo 'headless = true' >> ~/.streamlit/config.toml
echo 'enableCORS=false' >> ~/.streamlit/config.toml
#echo 'port = 8501' >> ~/.streamlit/config.toml
cat ~/.streamlit/credentials.toml
cat ~/.streamlit/config.toml


