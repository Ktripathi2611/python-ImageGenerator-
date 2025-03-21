import os
import sys
import streamlit as st
from PIL import Image
from datetime import datetime
import torch

# Disable PyTorch/Streamlit watchdog issue
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"

# Import generate_image
from generate_image import generate_image

# Page configuration
st.set_page_config(
    page_title="Stable Diffusion Generator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rest of your app.py code
# ... 