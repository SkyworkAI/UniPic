#!/usr/bin/env python3
import gradio as gr
import torch
import os
import sys
from PIL import Image

# Add project root to path to allow importing 'unipicv2'
# This is necessary because gradio.py is at the root level.
sys.path.append(os.path.abspath('.'))

# Import necessary components from the unipicv2 package and diffusers/transformers
from unipicv2.pipeline_stable_diffusion_3_kontext import StableDiffusion3KontextPipeline
from unipicv2.transformer_sd3_kontext import SD3Transformer2DKontextModel
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL, BitsAndBytesConfig
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

# --- Configuration ---
CHECKPOINT_PATH = "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DTYPE is now used mainly for the compute_dtype in the 4-bit config
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# --- Global Variable for the Pipeline ---
pipeline = None

# --- Model Loading Function (MODIFIED FOR INT4 WITH CPU OFFLOAD) ---
def load_model(checkpoint_path):
    """
    Loads all the necessary model components with int4 quantization and creates the pipeline.
    This function is called only once when the Gradio app starts.
    It now includes CPU offloading for systems with limited VRAM.
    """
    print(f"Loading model from: {checkpoint_path} using int4 quantization with CPU offload enabled...")

    try:
        # Define quantization configurations with CPU offloading enabled
        # This allows modules that don't fit on the GPU to be placed on the CPU RAM.
        
        # For the main transformer model (int4)
        bnb4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
            llm_int8_enable_fp32_cpu_offload=True  # <-- AJOUT IMPORTANT
        )
        
        # For the text encoders (int8)
        bnb8_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True  # <-- AJOUT IMPORTANT
        )

        # Load Transformer with int4 quantization
        transformer = SD3Transformer2DKontextModel.from_pretrained(
            checkpoint_path, 
            subfolder="transformer",
            quantization_config=bnb4_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        # VAE is loaded in float16 for stability
        vae = AutoencoderKL.from_pretrained(
            checkpoint_path, 
            subfolder="vae",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        # Load Text Encoders with int8 quantization
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            checkpoint_path, 
            subfolder="text_encoder",
            quantization_config=bnb8_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        tokenizer = CLIPTokenizer.from_pretrained(checkpoint_path, subfolder="tokenizer")

        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            checkpoint_path, 
            subfolder="text_encoder_2",
            quantization_config=bnb8_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(checkpoint_path, subfolder="tokenizer_2")

        text_encoder_3 = T5EncoderModel.from_pretrained(
            checkpoint_path, 
            subfolder="text_encoder_3",
            quantization_config=bnb8_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        tokenizer_3 = T5TokenizerFast.from_pretrained(checkpoint_path, subfolder="tokenizer_3")

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_path, subfolder="scheduler")

        # Create the pipeline object
        pipe = StableDiffusion3KontextPipeline(
            transformer=transformer, vae=vae,
            text_encoder=text_encoder, tokenizer=tokenizer,
            text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2,
            text_encoder_3=text_encoder_3, tokenizer_3=tokenizer_3,
            scheduler=scheduler
        )
        print("Model loaded successfully with CPU offloading.")
        return pipe
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have 'bitsandbytes' and 'accelerate' installed: pip install bitsandbytes accelerate")
        print(f"Also ensure the checkpoint exists at '{checkpoint_path}' and the file structure is correct.")
        return None



# --- Core Logic Functions (Adapted from your scripts) ---

def generate_text_to_image(prompt, negative_prompt, height, width, num_steps, guidance, seed):
    """
    Function for the Text-to-Image generation tab.
    """
    if pipeline is None:
        raise gr.Error("Model is not loaded. Please check the console for errors.")
        
    # The generator needs to be on the same device as the model. 'device_map' places it on cuda:0 by default.
    generator_device = pipeline.transformer.device
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    
    print(f"Generating image with prompt: '{prompt}'")
    
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=int(height), 
        width=int(width),
        num_inference_steps=int(num_steps),
        guidance_scale=guidance,
        generator=generator
    ).images[0]
    
    print("Image generation complete.")
    return image


def fix_longer_edge(x, image_size, factor=32):
    """Resize image while maintaining aspect ratio (from the editing script)."""
    w, h = x.size
    if w >= h:
        target_w = image_size
        target_h = h * (target_w / w)
        target_h = round(target_h / factor) * factor
    else:
        target_h = image_size
        target_w = w * (target_h / h)
        target_w = round(target_w / factor) * factor
    x = x.resize(size=(target_w, target_h))
    return x


def generate_image_editing(input_image, prompt, negative_prompt, num_steps, guidance, seed):
    """
    Function for the Image Editing tab.
    """
    if pipeline is None:
        raise gr.Error("Model is not loaded. Please check the console for errors.")
        
    if input_image is None:
        raise gr.Error("Please upload an input image for editing.")

    image = Image.fromarray(input_image)
    
    processed_image = fix_longer_edge(image, image_size=1024)
    print(f"Resized input image to: {processed_image.width}x{processed_image.height}")
    
    generator_device = pipeline.transformer.device
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    
    print(f"Editing image with prompt: '{prompt}'")

    edited_image = pipeline(
        image=processed_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=processed_image.height, 
        width=processed_image.width,
        num_inference_steps=int(num_steps),
        guidance_scale=guidance,
        generator=generator
    ).images[0]
    
    print("Image editing complete.")
    return edited_image

# --- Gradio Interface Definition ---

new_features_description = """
### What's New in UniPic2
- **Enhanced Architecture:** Improved model architecture based on Stable Diffusion 3.5 and Flux
- **Better Image Quality:** Advanced preprocessing and generation techniques
- **Unified Framework:** Seamless integration of text-to-image and image editing tasks
- **Optimized Inference:** More efficient inference pipeline with better memory management **(Now with INT4!)**
- **Advanced Parameters:** Fine-grained control over generation quality and style
"""

default_negative_prompt = "blurry, low quality, low resolution, distorted, deformed, broken content, missing parts, damaged details, artifacts, glitch, noise, pixelated, grainy, compression artifacts, bad composition, wrong proportion, incomplete editing, unfinished, unedited areas."

with gr.Blocks(theme='soft') as demo:
    gr.Markdown("# UniPic2: SD3.5M Kontext Playground")
    gr.Markdown(new_features_description)
    
    with gr.Tabs():
        with gr.TabItem("Text-to-Image Generation"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2i_prompt = gr.Textbox(label="Prompt", placeholder="A beautiful landscape painting", lines=3)
                    t2i_neg_prompt = gr.Textbox(label="Negative Prompt", placeholder="e.g., ugly, deformed", lines=2)
                    with gr.Row():
                        t2i_height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                        t2i_width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                    t2i_steps = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Inference Steps")
                    t2i_guidance = gr.Slider(minimum=1.0, maximum=10.0, value=3.5, step=0.1, label="Guidance Scale")
                    t2i_seed = gr.Number(label="Seed", value=42)
                    t2i_button = gr.Button("Generate", variant="primary")
                with gr.Column(scale=1):
                    t2i_output = gr.Image(label="Generated Image", type="pil")

        with gr.TabItem("Image Editing"):
            with gr.Row():
                with gr.Column(scale=1):
                    edit_input_image = gr.Image(type="numpy", label="Input Image")
                    edit_prompt = gr.Textbox(label="Editing Prompt", placeholder="Make the cat wear a wizard hat", lines=3)
                    edit_neg_prompt = gr.Textbox(label="Negative Prompt", value=default_negative_prompt, lines=2)
                    edit_steps = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Inference Steps")
                    edit_guidance = gr.Slider(minimum=1.0, maximum=10.0, value=3.5, step=0.1, label="Guidance Scale")
                    edit_seed = gr.Number(label="Seed", value=42)
                    edit_button = gr.Button("Edit Image", variant="primary")
                with gr.Column(scale=1):
                    edit_output = gr.Image(label="Edited Image", type="pil")

    t2i_button.click(
        fn=generate_text_to_image,
        inputs=[t2i_prompt, t2i_neg_prompt, t2i_height, t2i_width, t2i_steps, t2i_guidance, t2i_seed],
        outputs=t2i_output
    )
    
    edit_button.click(
        fn=generate_image_editing,
        inputs=[edit_input_image, edit_prompt, edit_neg_prompt, edit_steps, edit_guidance, edit_seed],
        outputs=edit_output
    )

if __name__ == "__main__":
    if os.path.exists(CHECKPOINT_PATH):
        pipeline = load_model(CHECKPOINT_PATH)
    else:
        print(f"ERROR: Checkpoint path not found at '{CHECKPOINT_PATH}'")
        print("The Gradio app will start, but model-related functions will fail.")
        print("Please ensure your 'models' directory is set up correctly.")

    demo.launch(share=True)