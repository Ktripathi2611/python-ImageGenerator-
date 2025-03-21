import torch
from diffusers import StableDiffusionPipeline
import argparse
from PIL import Image
import os
import gc

# Global variable to store the pipeline and avoid reloading
global_pipe = None

def generate_image(
    prompt,
    negative_prompt="",
    height=512,
    width=512,
    num_inference_steps=20,
    guidance_scale=7.5,
    output_path="output.png"
):
    global global_pipe
    
    try:
        # Try to use CUDA if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use existing pipeline if available
        if global_pipe is not None:
            pipe = global_pipe
        else:
            # Set low memory optimization parameters
            pipe_kwargs = {
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                "safety_checker": None,
                "requires_safety_checker": False,
            }
            
            # Load model with memory optimizations
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                **pipe_kwargs
            )
            
            # Move to device
            pipe = pipe.to(device)
            
            # Apply memory optimization techniques
            if device == "cuda":
                pipe.enable_attention_slicing()
            
            # Store for future use
            global_pipe = pipe
        
        # Explicitly clear CUDA cache
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        # Generate image with more controlled memory usage
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            ).images[0]
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Save the image and return
        image.save(output_path)
        print(f"Image saved to: {output_path}")
        
        # Clean up but don't delete the pipeline
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        return output_path
        
    except RuntimeError as e:
        # If CUDA error occurs, try again with CPU
        if "CUDA" in str(e) and device == "cuda":
            print("CUDA error encountered. Falling back to CPU...")
            torch.cuda.empty_cache()
            gc.collect()
            
            # Reset global pipeline
            global_pipe = None
            
            # Create a clean pipeline on CPU
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            pipe = pipe.to("cpu")
            
            # Store for future use
            global_pipe = pipe
            
            # Generate with CPU (will be slower)
            with torch.no_grad():
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                ).images[0]
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            
            image.save(output_path)
            print(f"Image saved to: {output_path} (CPU mode)")
            
            gc.collect()
            
            return output_path
        else:
            # Re-raise if it's not a CUDA error or if CPU also failed
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image using Stable Diffusion")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative text prompt")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--output", type=str, default="output.png", help="Output file path")
    
    args = parser.parse_args()
    
    generate_image(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        output_path=args.output
    ) 