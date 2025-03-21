import os
import subprocess
import sys

# Disable Streamlit's file watcher (this is causing the PyTorch class error)
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# Launch the Streamlit app
subprocess.run([
    sys.executable, "-m", "streamlit", "run", 
    "app.py", 
    "--server.runOnSave=false",
    "--server.fileWatcherType=none"
]) 