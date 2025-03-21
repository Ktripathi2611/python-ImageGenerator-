
# Stable Diffusion Image Generator

A streamlined web interface for generating AI images using Stable Diffusion, optimized for computers with limited GPU memory (4GB+ VRAM). Features hybrid CPU+GPU processing for optimal performance on lower-end hardware.

## 🌟 Features

- **Low VRAM Optimization**: Works on GPUs with as little as 4GB VRAM
- **Hybrid Processing**: Seamlessly switches between GPU and CPU
- **User-friendly Interface**: Clean, intuitive Streamlit web UI
- **Smart Memory Management**: Automatic memory optimization and cleanup
- **Built-in Gallery**: View and manage your generated images
- **Prompt Engineering Guide**: Learn to write effective prompts

## 🛠️ Installation

### Prerequisites

- Python 3.8-3.11 (3.12 not supported yet)
- CUDA-compatible GPU (optional)
- 4GB+ VRAM (for GPU processing)
- 8GB+ RAM

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/stable-diffusion-generator.git
cd stable-diffusion-generator

# Install dependencies
pip install -r requirements.txt

# Run the application
python run_app.py
```

## 📝 Project Structure

```
stable-diffusion-generator/
├── app.py              # Main Streamlit interface
├── generate_image.py   # Core image generation logic
├── run_app.py          # Launcher script
├── app_deploy.py       # Deployment version
├── requirements.txt    # Python dependencies
├── Dockerfile         # Container configuration
├── .streamlit/        # Streamlit configuration
│   └── config.toml
└── outputs/           # Generated images directory
```

## 🚀 Usage

1. **Start the Application**:

   ```bash
   python run_app.py
   ```

   Access the interface at http://localhost:8501

2. **Generate Images**:

   - Enter your prompt
   - Adjust settings (optional)
   - Click "Generate Image"
   - Download or view in gallery

3. **Command Line Usage**:
   ```bash
   python generate_image.py --prompt "Your prompt here" --width 512 --height 512
   ```

## 🌐 Deployment Options

### 1. Docker

```bash
# Build image
docker build -t stable-diffusion-app .

# Run container
docker run -p 8501:8501 stable-diffusion-app
```

### 2. Streamlit Cloud

- Push to GitHub
- Connect to [streamlit.io/cloud](https://streamlit.io/cloud)
- Select app.py as main file

### 3. Hugging Face Spaces

- Create Space on [huggingface.co](https://huggingface.co)
- Choose Streamlit as Space SDK
- Upload project files

## ⚙️ Memory Optimization

Built-in optimizations for low VRAM usage:

- Half-precision (FP16) computation
- Attention slicing
- Pipeline caching
- Dynamic CPU offloading
- Aggressive memory cleanup

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   - Reduce image dimensions
   - Lower inference steps
   - Close other GPU applications
   - Will automatically try CPU fallback

2. **Module Import Errors**

   ```bash
   pip install -r requirements.txt
   ```

3. **Slow Generation**
   - CPU mode is significantly slower
   - Reduce inference steps
   - Check GPU availability in System Info

## 📊 Parameters Guide

### Inference Steps (10-50)

- **Low (10-15)**: Fast, basic quality
- **Medium (20-30)**: Balanced (recommended)
- **High (40-50)**: Best quality, slower

### Guidance Scale (1-15)

- **Low (1-5)**: Creative, abstract
- **Medium (7-9)**: Balanced
- **High (10-15)**: Literal, precise

## 📜 License

This project is available under the MIT License.

## 🙏 Acknowledgments

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Streamlit](https://streamlit.io)

---

Created by [Your Name] - For AI image generation on limited hardware
