import importlib
import sys
import streamlit as st
fnames = ["app_sentiment.py","app.py"]
fname_to_run = st.sidebar.selectbox('Select an app', fnames)

# Create module from filepath and put in sys.modules, so Streamlit knows
# to watch it for changes.

fake_module_count = 0

def load_module(filepath):
    global fake_module_count
    modulename = '_dont_care_%s' % fake_module_count
    spec = importlib.util.spec_from_file_location(modulename, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[modulename] = module

    fake_module_count += 1

# Run the selected file.

with open(fname_to_run) as f:
    load_module(fname_to_run)
    filebody = f.read()
exec(filebody, {})