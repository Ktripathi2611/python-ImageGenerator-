<think>
I need to create a comprehensive README file for this Stable Diffusion project. The README should explain:

1. What the project is (Stable Diffusion image generator optimized for low VRAM GPUs)
2. How to set it up and install dependencies
3. How to run the application
4. How the application works
5. Tips for using it effectively
6. Troubleshooting common issues
7. Technical details about the optimizations for low VRAM

The project consists of:
- generate_image.py - The core function that handles image generation
- app.py - The Streamlit web interface
- run_app.py - Launcher that fixes PyTorch/Streamlit compatibility issues

I'll structure the README to be clear and include markdown formatting.
</think>

# Stable Diffusion Image Generator for Low VRAM GPUs

This project provides a user-friendly interface for generating AI images with Stable Diffusion, specifically optimized for computers with limited GPU memory (like your 4GB RTX 2050). It uses a hybrid CPU+GPU approach to allow image generation even on systems with low VRAM.

![Stable Diffusion Generator](https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/imgs/stable_diffusion.png)

## 📑 Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Optimizations](#optimizations)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)

## ✨ Features

- **Low VRAM Optimization**: Works on GPUs with as little as 4GB VRAM
- **Automatic CPU Fallback**: If GPU memory is exhausted, falls back to CPU
- **User-friendly Interface**: Easy-to-use Streamlit web interface
- **Example Prompts**: Built-in examples to get you started
- **Image Gallery**: View all your previously generated images
- **Prompt Engineering Tips**: Learn how to write better prompts
- **Memory Usage Stats**: Monitor your GPU memory usage

## 🖥️ Requirements

- Python 3.8-3.11 (3.12 has compatibility issues with PyTorch and Streamlit)
- CUDA-compatible GPU (optional, but recommended)
- At least 4GB of VRAM (if using GPU)
- 8GB+ of RAM

## 🔧 Installation

1. **Clone or download this repository**

2. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   pip install diffusers transformers accelerate streamlit pillow numpy
   ```

3. **Verify GPU access** (optional)
   ```python
   import torch
   print(torch.cuda.is_available())  # Should return True if GPU is detected
   ```

## 🚀 Usage

### Running the Application

To start the application:

```bash
python run_app.py
```

> ⚠️ Note: Always use the `run_app.py` launcher instead of directly running `streamlit run app.py`. The launcher fixes compatibility issues between PyTorch and Streamlit.

### Using the Interface

1. **Generate Tab**: Create new images
   - Enter a prompt describing what you want to see
   - (Optional) Expand "Advanced Options" to:
     - Add negative prompts (what you don't want to see)
     - Adjust image dimensions
     - Change inference steps and guidance scale
   - Click "Generate Image" button

2. **Gallery Tab**: View your creations
   - Browse all previously generated images
   - View prompt details for each image
   - Download individual images

3. **Tips Tab**: Learn prompt techniques
   - Prompt structure guidance
   - Example prompts for different scenarios
   - Optimization recommendations

### Example Prompts

Use the example prompt selector in the sidebar to quickly try these prompts:

- **Fantasy Landscape**: "A magical fantasy landscape with floating islands, waterfalls, and ancient ruins, highly detailed digital painting, trending on artstation"
- **Cyberpunk City**: "A cyberpunk cityscape at night with neon lights, rain, and flying cars, cinematic lighting, detailed, 8k"
- **Portrait**: "Portrait of a young woman with blue eyes, photorealistic, studio lighting, detailed skin, professional photography"
- **Sci-fi Scene**: "A futuristic space station orbiting Jupiter, astronauts working outside, detailed, realistic, cinematic"

## 🔍 How It Works

### Project Structure

This project consists of three main files:

1. **generate_image.py**: Core image generation function
   - Handles the Stable Diffusion model
   - Implements memory optimizations
   - Provides automatic CPU fallback

2. **app.py**: Streamlit web interface
   - Provides user-friendly UI for generating images
   - Displays generated images in a gallery
   - Shows system information and tips

3. **run_app.py**: Launcher script
   - Fixes compatibility issues between PyTorch and Streamlit
   - Disables file watchers that cause errors

### Image Generation Pipeline

1. **Model Loading**: Loads Stable Diffusion 1.5 with optimizations
2. **Text Processing**: Converts your prompt to embeddings
3. **Denoising**: Gradually removes noise from a random seed
4. **Image Creation**: Converts denoised latent to final image
5. **Save & Display**: Saves the image and displays in the UI

## 🔧 Optimizations

This project employs several optimizations to work with limited VRAM:

1. **Half-precision (FP16)**: Uses 16-bit floating point on GPU to reduce memory usage
2. **Attention Slicing**: Processes attention mechanisms in smaller chunks
3. **Safety Checker Disabled**: Removes unnecessary memory usage
4. **Automatic Memory Management**: Clears CUDA cache after generation
5. **CPU Offloading**: Falls back to CPU when GPU memory is exhausted
6. **Efficient Component Loading**: Only loads components when needed

## ❓ Troubleshooting

### Common Issues

1. **"CUDA out of memory" error**
   - Reduce image dimensions (try 384x384)
   - Lower the number of inference steps
   - Close other applications using the GPU
   - The app will automatically try CPU fallback

2. **"No module named 'X'" error**
   - Make sure you've installed all dependencies:
     ```bash
     pip install torch diffusers transformers accelerate streamlit pillow numpy
     ```

3. **"RuntimeError: no running event loop" or PyTorch class errors**
   - Use `python run_app.py` instead of `streamlit run app.py`
   - If still occurring, try Python 3.10 instead of 3.12

4. **Very slow generation**
   - If running on CPU, this is expected (10+ minutes per image)
   - Check if CUDA is available with the System Information panel
   - Try fewer inference steps (15-20 is often sufficient)

### Performance Tips

1. **Optimal Settings for 4GB VRAM**:
   - Resolution: 448x448 or 512x512
   - Steps: 20-30
   - Make sure to enable attention slicing

2. **Quality vs. Speed**:
   - More steps = better quality but slower generation
   - Higher guidance scale = more prompt adherence but less variety

## 🔄 Advanced Usage

### Command Line Generation

You can also generate images directly from the command line:

```bash
python generate_image.py --prompt "Your detailed prompt here" --negative_prompt "Things to avoid" --width 512 --height 512 --steps 30 --guidance 7.5 --output output.png
```

### Customizing the Model

To use a different model, modify the `generate_image.py` file:

```python
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",  # Change to another model
    torch_dtype=torch.float16
)
```

Recommended alternative models:
- `"runwayml/stable-diffusion-v1-5"` (default)
- `"CompVis/stable-diffusion-v1-4"`
- `"stabilityai/stable-diffusion-2-1"`

Note that larger models may require more VRAM.

## 📝 Technical Details

### File Descriptions

- **generate_image.py**: Core image generation function implementing Stable Diffusion with memory optimizations.
- **app.py**: Streamlit web interface providing a user-friendly front-end.
- **run_app.py**: Launcher script that fixes PyTorch/Streamlit compatibility issues.
- **outputs/**: Directory where generated images and their prompts are saved.

### Memory Optimization Techniques

The key memory optimizations in this project include:

```python
# Use half precision to save VRAM
pipe_kwargs = {
    "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    "safety_checker": None,  # Disable to save memory
    "requires_safety_checker": False,
}

# Slice attention computation into smaller chunks
pipe.enable_attention_slicing()

# Explicitly clear CUDA cache
torch.cuda.empty_cache()
gc.collect()
```

## 🙏 Acknowledgements

This project utilizes:
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) by CompVis
- [Streamlit](https://streamlit.io/) for the web interface


---

Created for users with limited GPU resources to enjoy the magic of AI image generation.
