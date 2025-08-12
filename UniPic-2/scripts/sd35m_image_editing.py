#!/usr/bin/env python3
"""
Skywork-UniPic2 SD3.5M Kontext Image Editing
Usage: python sd35m_image_editing.py --checkpoint_path /path/to/checkpoint --input_image input.png --prompt "edit description" --output output.png
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import argparse
from PIL import Image
from unipicv2.pipeline_stable_diffusion_3_kontext import StableDiffusion3KontextPipeline
from unipicv2.transformer_sd3_kontext import SD3Transformer2DKontextModel
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast


def fix_longer_edge(x, image_size, factor=32):
    """Resize image while maintaining aspect ratio"""
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


def main():
    parser = argparse.ArgumentParser(description='SD3.5M Kontext Image Editing')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to the unipicv2_sd_3_5m_kontext checkpoint')
    parser.add_argument('--input_image', type=str, required=True,
                      help='Path to input image for editing')
    parser.add_argument('--prompt', type=str, required=True,
                      help='Text prompt for image editing')
    parser.add_argument('--negative_prompt', type=str, 
                      default="blurry, low quality, low resolution, distorted, deformed, broken content, missing parts, damaged details, artifacts, glitch, noise, pixelated, grainy, compression artifacts, bad composition, wrong proportion, incomplete editing, unfinished, unedited areas.",
                      help='Negative prompt for image editing')
    parser.add_argument('--output', type=str, default='image_editing.png',
                      help='Output image path')
    parser.add_argument('--image_size', type=int, default=512,
                      help='Target image size for resizing')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                      help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=3.5,
                      help='Guidance scale')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.checkpoint_path}")
    
    # Load model components
    transformer = SD3Transformer2DKontextModel.from_pretrained(
        args.checkpoint_path, subfolder="transformer", torch_dtype=torch.bfloat16).cuda()

    vae = AutoencoderKL.from_pretrained(
        args.checkpoint_path, subfolder="vae", torch_dtype=torch.bfloat16).cuda()

    # Load text encoders
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        args.checkpoint_path, subfolder="text_encoder", torch_dtype=torch.bfloat16).cuda()
    tokenizer = CLIPTokenizer.from_pretrained(args.checkpoint_path, subfolder="tokenizer")

    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.checkpoint_path, subfolder="text_encoder_2", torch_dtype=torch.bfloat16).cuda()
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.checkpoint_path, subfolder="tokenizer_2")

    text_encoder_3 = T5EncoderModel.from_pretrained(
        args.checkpoint_path, subfolder="text_encoder_3", torch_dtype=torch.bfloat16).cuda()
    tokenizer_3 = T5TokenizerFast.from_pretrained(args.checkpoint_path, subfolder="tokenizer_3")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.checkpoint_path, subfolder="scheduler")

    # Create pipeline
    pipeline = StableDiffusion3KontextPipeline(
        transformer=transformer, vae=vae,
        text_encoder=text_encoder, tokenizer=tokenizer,
        text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2,
        text_encoder_3=text_encoder_3, tokenizer_3=tokenizer_3,
        scheduler=scheduler)

    # Load and preprocess image
    print(f"Loading input image: {args.input_image}")
    image = Image.open(args.input_image)
    image = fix_longer_edge(image, image_size=args.image_size)
    print(f"Resized image to: {image.width}x{image.height}")

    print(f"Editing image with prompt: '{args.prompt}'")
    
    # Edit image
    edited_image = pipeline(
        image=image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=image.height, width=image.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device=transformer.device).manual_seed(args.seed)
    ).images[0]

    edited_image.save(args.output)
    print(f"Edited image saved to: {args.output}")


if __name__ == "__main__":
    main()