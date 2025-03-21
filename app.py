import os
import sys
import streamlit as st
from PIL import Image
from datetime import datetime
import torch

# Disable PyTorch/Streamlit watchdog issue
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"

# Import generate_image in a way that prevents Streamlit from scanning PyTorch classes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_image import generate_image

# Page configuration
st.set_page_config(
    page_title="Stable Diffusion Generator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create outputs directory if it doesn't exist
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Initialize session state for example prompts
if "example_prompt" not in st.session_state:
    st.session_state.example_prompt = ""

# Sidebar with instructions
with st.sidebar:
    st.title("About This App")
    st.markdown("""
    This is a **Stable Diffusion** image generator optimized for computers with limited GPU memory.
    
    ### How it works:
    1. Enter a detailed prompt describing what you want to see
    2. Add negative prompts to avoid unwanted elements
    3. Adjust settings as needed
    4. Click "Generate Image"
    """)
    
    # Example prompts
    st.subheader("Example Prompts")
    example_prompts = {
        "Fantasy Landscape": "A magical fantasy landscape with floating islands, waterfalls, and ancient ruins, highly detailed digital painting, trending on artstation",
        "Cyberpunk City": "A cyberpunk cityscape at night with neon lights, rain, and flying cars, cinematic lighting, detailed, 8k",
        "Portrait": "Portrait of a young woman with blue eyes, photorealistic, studio lighting, detailed skin, professional photography",
        "Sci-fi Scene": "A futuristic space station orbiting Jupiter, astronauts working outside, detailed, realistic, cinematic"
    }
    
    selected_example = st.selectbox("Try an example:", [""] + list(example_prompts.keys()))
    
    if selected_example and st.button("Use This Example"):
        st.session_state.example_prompt = example_prompts[selected_example]
    
    # System information
    st.subheader("System Information")
    st.write(f"Python: {sys.version.split()[0]}")
    st.write(f"PyTorch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    st.write(f"CUDA: {'✓ Available' if cuda_available else '✗ Not available'}")
    if cuda_available:
        st.write(f"GPU: {torch.cuda.get_device_name(0)}")
        st.write(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# Main content
st.title("Stable Diffusion Image Generator")
st.caption("Optimized for Low VRAM GPUs")

# Create tabs for different functions
tab1, tab2, tab3 = st.tabs(["Generate Image", "Gallery", "Tips"])

# Tab 1: Image Generation
with tab1:
    # Form for input collection
    with st.form(key="generation_form"):
        prompt = st.text_area(
            "Enter your prompt:", 
            height=100,
            value=st.session_state.example_prompt,
            help="Describe what you want to see in the image"
        )
        
        # Clear example prompt after use
        if st.session_state.example_prompt:
            st.session_state.example_prompt = ""
        
        # Advanced options in expander
        with st.expander("Advanced Options"):
            negative_prompt = st.text_area(
                "Negative prompt:", 
                value="ugly, bad anatomy, blurry, pixelated, deformed, low quality, text, watermark",
                height=100
            )
            
            # Image dimensions
            st.markdown("#### Image Dimensions")
            col1, col2 = st.columns(2)
            with col1:
                width = st.select_slider(
                    "Width:", 
                    options=[384, 448, 512, 576, 640], 
                    value=512
                )
            with col2:
                height = st.select_slider(
                    "Height:", 
                    options=[384, 448, 512, 576, 640], 
                    value=512
                )
            
            # Generation parameters
            st.markdown("#### Generation Parameters")
            st.markdown("""
            <small>These parameters control the quality and style of the generated image:</small>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                num_inference_steps = st.slider(
                    "Inference Steps:", 
                    min_value=10, 
                    max_value=50, 
                    value=20,
                    help="Controls the number of denoising steps. More steps = better quality but slower generation. 20-30 steps is a good balance for most images."
                )
                st.markdown("""
                <small>**Inference Steps** determine how many iterations the AI takes to create your image. 
                Think of it like a painter adding more and more detail with each pass:</small>
                """, unsafe_allow_html=True)
                st.markdown("""
                <small>• Low (10-15): Faster but less detailed</small>
                """, unsafe_allow_html=True)
                st.markdown("""
                <small>• Medium (20-30): Good balance for most images</small>
                """, unsafe_allow_html=True)
                st.markdown("""
                <small>• High (40-50): Maximum detail but slower generation</small>
                """, unsafe_allow_html=True)
            
            with col2:
                guidance_scale = st.slider(
                    "Guidance Scale:", 
                    min_value=1.0, 
                    max_value=15.0, 
                    value=7.5, 
                    step=0.5,
                    help="Controls how closely the image follows your prompt. Higher values = more literal interpretation but potentially less creative results."
                )
                st.markdown("""
                <small>**Guidance Scale** controls how closely the AI follows your text prompt:</small>
                """, unsafe_allow_html=True)
                st.markdown("""
                <small>• Low (1-5): More creative/abstract interpretations</small>
                """, unsafe_allow_html=True)
                st.markdown("""
                <small>• Medium (7-9): Balanced adherence to prompt</small>
                """, unsafe_allow_html=True)
                st.markdown("""
                <small>• High (10-15): Very literal interpretation, but may cause oversaturation or artifacts</small>
                """, unsafe_allow_html=True)
        
        # Filename and checkboxes
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"image_{timestamp}.png")
        
        save_prompt = st.checkbox("Save prompt with image", value=True)
        
        # Submit button
        generate_button = st.form_submit_button("Generate Image")
    
    # Processing (outside the form)
    if generate_button:
        if not prompt:
            st.error("Please enter a prompt!")
        else:
            # Progress display
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status = st.empty()
                
                status.text("Initializing model...")
                progress_bar.progress(10)
                
                try:
                    # Update progress
                    status.text("Generating image... (this may take a while)")
                    progress_bar.progress(30)
                    
                    # Generate image
                    image_path = generate_image(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        output_path=output_path
                    )
                    
                    # Save prompt if requested
                    if save_prompt and image_path:
                        prompt_file = image_path.replace(".png", "_prompt.txt")
                        with open(prompt_file, "w") as f:
                            f.write(f"Prompt: {prompt}\n\n")
                            f.write(f"Negative prompt: {negative_prompt}\n\n")
                            f.write(f"Settings: Steps={num_inference_steps}, Guidance={guidance_scale}, Size={width}x{height}")
                    
                    # Finalize progress
                    progress_bar.progress(100)
                    status.text("Image generation complete!")
                    
                    # Display result
                    if image_path and os.path.exists(image_path):
                        progress_container.empty()
                        st.success("Image generated successfully!")
                        
                        # Show image
                        st.image(Image.open(image_path), caption=f"Generated image ({width}x{height})")
                        
                        # Prompt details
                        with st.expander("View prompt details"):
                            st.markdown(f"**Prompt:**\n{prompt}")
                            st.markdown(f"**Negative prompt:**\n{negative_prompt}")
                            st.markdown(f"**Settings:** Steps={num_inference_steps}, Guidance={guidance_scale}, Size={width}x{height}")
                        
                        # Download section
                        download_col1, download_col2 = st.columns(2)
                        with download_col1:
                            with open(image_path, "rb") as file:
                                st.download_button(
                                    label="Download Image",
                                    data=file,
                                    file_name=os.path.basename(image_path),
                                    mime="image/png",
                                    use_container_width=True
                                )
                    else:
                        st.error("Failed to generate image. Check console for errors.")
                
                except Exception as e:
                    progress_container.empty()
                    st.error(f"Error during image generation: {str(e)}")
                    st.info("If you're seeing memory errors, try reducing image dimensions or inference steps.")

# Tab 2: Gallery
with tab2:
    st.subheader("Image Gallery")
    
    # Check for images
    if os.path.exists(output_dir):
        image_files = [f for f in os.listdir(output_dir) if f.endswith('.png') and not f.endswith('_prompt.png')]
        image_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
        
        if image_files:
            st.write(f"Showing {len(image_files)} generated images:")
            
            # Display grid
            cols = 3
            for i in range(0, len(image_files), cols):
                image_row = st.columns(cols)
                
                for j in range(cols):
                    idx = i + j
                    if idx < len(image_files):
                        image_file = image_files[idx]
                        img_path = os.path.join(output_dir, image_file)
                        
                        # Find prompt file
                        prompt_file = img_path.replace(".png", "_prompt.txt")
                        prompt_text = ""
                        if os.path.exists(prompt_file):
                            with open(prompt_file, "r") as f:
                                prompt_text = f.read()
                        
                        # Display in column
                        with image_row[j]:
                            st.image(Image.open(img_path), use_container_width=True)
                            
                            if prompt_text:
                                with st.expander("View details"):
                                    st.text(prompt_text)
                            
                            # Create a unique key for each download button
                            with open(img_path, "rb") as file:
                                st.download_button(
                                    label="Download",
                                    data=file,
                                    file_name=image_file,
                                    mime="image/png",
                                    key=f"download_{idx}_{image_file}"
                                )
        else:
            st.info("No images found in the gallery.")
    else:
        st.info("No images have been generated yet.")

# Tab 3: Tips
with tab3:
    st.subheader("Prompt Engineering Tips")
    
    st.markdown("""
    ## 🎨 Effective Prompts
    
    ### Structure your prompts:
    ```
    [Type of image] of [subject], [descriptive details], [art style], [lighting], [other parameters]
    ```
    
    ### Examples:
    - "Portrait of a young woman with blue eyes, blonde hair, detailed features, professional photography, soft lighting, bokeh background, 4k"
    - "Landscape photo of mountains at sunset, dramatic sky, golden hour, cinematic, detailed, sharp focus, 8k"
    
    ### Effective Modifiers:
    - **Quality boosters**: highly detailed, sharp focus, 8k, high resolution
    - **Style references**: trending on artstation, cinematic, photorealistic
    - **Artists**: in the style of [artist name]
    - **Camera settings**: bokeh, shallow depth of field, telephoto lens
    
    ## 🛠️ Optimization Tips
    
    ### If you're running out of memory:
    1. Reduce image dimensions (384x384 needs less VRAM)
    2. Lower the number of inference steps (20 is often sufficient)
    3. Close other applications using GPU
    4. The app will automatically fallback to CPU if needed (but will be much slower)
    """)

    st.markdown("""
    ## 🎛️ Understanding Parameters

    ### Inference Steps (10-50)
    This controls how many denoising iterations the AI performs when creating your image:

    - **How it works**: Stable Diffusion starts with random noise and gradually refines it into an image over multiple steps
    - **Low steps (10-15)**: Faster generation, less detail, may have some noise/artifacts
    - **Medium steps (20-30)**: Good balance between speed and quality for most images
    - **High steps (40-50)**: Maximum detail and coherence, but significantly slower

    When to adjust: Lower this if generation is too slow; increase if you notice poor details or artifacts.

    ### Guidance Scale (1-15)
    This controls how strictly the AI adheres to your text prompt:

    - **How it works**: Controls the balance between following your prompt exactly versus allowing creative interpretation
    - **Low values (1-5)**: More creative, abstract, and sometimes unpredictable results
    - **Medium values (7-9)**: Balanced adherence to prompt while maintaining quality
    - **High values (10-15)**: Very literal interpretation of your prompt, but may cause:
      - Oversaturation of colors
      - Exaggerated features
      - Unnatural compositions or artifacts

    When to adjust: Higher for portraits or specific scenes you want to control precisely; lower for artistic or abstract interpretations.

    ### The Relationship Between Parameters

    - **High guidance + high steps**: Maximum prompt adherence and detail, slowest generation
    - **Low guidance + low steps**: Quick, creative results that may loosely interpret your prompt
    - **High guidance + low steps**: Fast generation that follows prompt but may have artifacts
    - **Low guidance + high steps**: Detailed, polished images with creative interpretation of your prompt
    """)

    st.subheader("Parameter Visualization")

    # Create a simple 2x2 grid showing the effect of parameters
    param_cols = st.columns(2)

    with param_cols[0]:
        st.markdown("##### Low Guidance Scale (3.0)")
        st.markdown("*Creative, interpretive results*")
        
    with param_cols[1]:
        st.markdown("##### High Guidance Scale (12.0)")
        st.markdown("*Literal, precise adherence to prompt*")

    st.markdown("---")

    st.write("Experiment with different combinations to find what works best for your specific prompts!") 