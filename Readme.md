<think>
I need to update the README.md file to explain the project in detail, including all components and deployment options. The README should cover:

1. Project overview
2. File structure explanation
3. Installation instructions
4. Usage instructions
5. Deployment options
6. Optimization techniques
7. Troubleshooting

I'll create a comprehensive README that includes all the necessary information for someone to understand, use, and deploy the project.
</think>

# Stable Diffusion Image Generator

This project provides a user-friendly web interface for generating AI images using Stable Diffusion, specifically optimized for computers with limited GPU memory. It employs a hybrid CPU+GPU approach to enable image generation even on systems with as little as 4GB VRAM.

![Stable Diffusion Example](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/stable_diffusion_1_5_example.png)

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment Options](#deployment-options)
- [Memory Optimization](#memory-optimization)
- [Troubleshooting](#troubleshooting)

## Features

- **Low VRAM Optimization**: Works on GPUs with as little as 4GB VRAM
- **Hybrid Processing**: Utilizes both GPU and CPU for optimal performance
- **User-friendly Interface**: Intuitive Streamlit web interface
- **Example Prompts**: Built-in examples to help you get started
- **Image Gallery**: Browse and download your previously generated images
- **Prompt Engineering Tips**: Learn how to write better prompts
- **Parameter Explanations**: Detailed guidance on inference steps and guidance scale
- **Automatic CPU Fallback**: Falls back to CPU processing if GPU memory is exhausted

## Project Structure

The project consists of the following files:

- `generate_image.py`: Core image generation function with memory optimizations
- `app.py`: Main Streamlit web interface
- `run_app.py`: Launcher script that fixes PyTorch/Streamlit compatibility issues
- `app_deploy.py`: Deployment-ready version of the app
- `requirements.txt`: Python dependencies
- `.streamlit/config.toml`: Streamlit configuration
- `Dockerfile`: Container configuration for Docker deployment
- `outputs/`: Directory where generated images are saved

## Installation

### Prerequisites

- Python 3.8-3.11 (3.12 has compatibility issues with PyTorch and Streamlit)
- CUDA-compatible GPU (optional, but recommended)
- At least 4GB of VRAM (if using GPU)
- 8GB+ of RAM

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/stable-diffusion-generator.git
cd stable-diffusion-generator
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify GPU Access (Optional)

```python
import torch
print(torch.cuda.is_available())  # Should return True if GPU is detected
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

## Usage

### Running the Application

Always use the launcher script to avoid compatibility issues:

```bash
python run_app.py
```

This will start the Streamlit web interface on http://localhost:8501

### Using the Interface

1. **Generate Tab**:
   - Enter a prompt describing what you want to see
   - (Optional) Use advanced options to customize:
     - Negative prompts (what you don't want to see)
     - Image dimensions (384-640px)
     - Inference steps (10-50)
     - Guidance scale (1-15)
   - Click "Generate Image"

2. **Gallery Tab**:
   - Browse all your previously generated images
   - View prompt details for each image
   - Download any image with one click

3. **Tips Tab**:
   - Learn prompt structure techniques
   - Understand parameter effects
   - Find optimization recommendations

### Command Line Usage

You can also generate images directly from the command line:

```bash
python generate_image.py --prompt "Your detailed prompt here" --negative_prompt "Things to avoid" --width 512 --height 512 --steps 30 --guidance 7.5 --output output.png
```

## Deployment Options

This project can be deployed in several ways:

### 1. Streamlit Cloud (Easiest)

1. Push your code to GitHub
2. Sign up at [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Select `app.py` as the main file

Streamlit Cloud provides free hosting with some resource limitations.

### 2. Hugging Face Spaces

1. Sign up at [huggingface.co](https://huggingface.co)
2. Create a new Space (Streamlit type)
3. Upload your files or connect to GitHub

Hugging Face Spaces offers free GPU runtime for demos.

### 3. Docker Deployment

For more control and consistent environments:

```bash
# Build the Docker image
docker build -t stable-diffusion-app .

# Run the container
docker run -p 8501:8501 stable-diffusion-app
```

This will make the app available at http://localhost:8501

### 4. Local Machine as Server

For quick deployment on your own hardware:

```bash
streamlit run app_deploy.py --server.address=0.0.0.0 --server.port=8501
```

This makes the app accessible to other devices on your network.

## Memory Optimization

This application implements several techniques to minimize GPU memory usage:

1. **Half-precision (FP16)**: Uses 16-bit floating point on GPU to reduce memory usage by ~50%
2. **Attention Slicing**: Processes the attention mechanism in smaller chunks
3. **Pipeline Caching**: Keeps the model loaded between generations
4. **Dynamic Offloading**: Falls back to CPU when GPU memory is exhausted
5. **Garbage Collection**: Aggressively frees memory after processing

These optimizations allow the application to work on GPUs with limited VRAM (4GB+) while maintaining reasonable generation speed.

## Troubleshooting

### Common Issues

1. **"CUDA out of memory" error**
   - Reduce image dimensions (try 384x384)
   - Lower the number of inference steps
   - Close other applications using GPU memory
   - The app will automatically try CPU fallback

2. **"No module named X" error**
   - Make sure you've installed all dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **PyTorch/Streamlit compatibility issues**
   - Always use `python run_app.py` instead of `streamlit run app.py`
   - If you encounter `setIn` errors or other Streamlit UI issues, try restarting the app
   - Python 3.10 is recommended for best compatibility

4. **Very slow generation**
   - If running on CPU, this is expected (5-15 minutes per image)
   - Check if CUDA is available in the System Information panel
   - Try fewer inference steps (15-20 is often sufficient)

### Performance Tips

1. **Optimal Settings for 4GB VRAM**:
   - Resolution: 384x384 or 448x448
   - Steps: 20-25
   - Make sure you have no other GPU-intensive applications running

2. **Quality vs. Speed Trade-offs**:
   - More inference steps = better quality but slower generation
   - Higher guidance scale = more prompt adherence but less variety
   - Larger dimensions = more detailed images but higher memory usage

---

Created for AI image generation on computers with limited resources. Enjoy creating with Stable Diffusion!

For issues, suggestions, or contributions, please open an issue or pull request on the repository.
