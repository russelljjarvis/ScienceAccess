install_dir="."
json=$(curl -s https://api.github.com/repos/mozilla/geckodriver/releases/latest)
url=$(echo "$json" | jq -r '.assets[].browser_download_url | select(contains("linux64"))')
curl -s -L "$url" | tar -xz
chmod +x geckodriver
sudo mv geckodriver "$install_dir"
echo "installed geckodriver binary in $install_dir"

web: sh setup.sh && streamlit run app.py
