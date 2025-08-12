#!/usr/bin/env python3
"""
Skywork-UniPic2 Qwen2.5-VL + SD3.5M Kontext Image Editing with Vision Input
Usage: python qwen_image_editing.py --checkpoint_path /path/to/checkpoint --input_image input.png --prompt "edit description" --output output.png
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from PIL import Image
from unipicv2.pipeline_stable_diffusion_3_kontext import StableDiffusion3KontextPipeline
from unipicv2.transformer_sd3_kontext import SD3Transformer2DKontextModel
from unipicv2.stable_diffusion_3_conditioner import StableDiffusion3Conditioner
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL


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
    parser = argparse.ArgumentParser(description='Qwen2.5-VL + SD3.5M Kontext Image Editing')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to the unipicv2_qwen2_5_vl_7b_sd_3_5m_kontext checkpoint')
    parser.add_argument('--input_image', type=str, required=True,
                      help='Path to input image for editing')
    parser.add_argument('--prompt', type=str, required=True,
                      help='Text prompt for image editing')
    parser.add_argument('--negative_prompt', type=str, 
                      default="blurry, low quality, low resolution, distorted, deformed, broken content, missing parts, damaged details, artifacts, glitch, noise, pixelated, grainy, compression artifacts, bad composition, wrong proportion, incomplete editing, unfinished, unedited areas.",
                      help='Negative prompt for image editing')
    parser.add_argument('--output', type=str, default='qwen_image_editing.png',
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

    # Load Qwen2.5-VL model
    print("Loading Qwen2.5-VL model...")
    lmm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/mnt/datasets_vlm/models/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").cuda()

    processor = Qwen2_5_VLProcessor.from_pretrained("/mnt/datasets_vlm/models/Qwen2.5-VL-7B-Instruct")
    processor.chat_template = processor.chat_template.replace(
        "{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}",
        "")

    conditioner = StableDiffusion3Conditioner.from_pretrained(
        args.checkpoint_path, subfolder="conditioner", torch_dtype=torch.bfloat16).cuda()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.checkpoint_path, subfolder="scheduler")

    # Create pipeline (note: text encoders set to None)
    pipeline = StableDiffusion3KontextPipeline(
        transformer=transformer, vae=vae,
        text_encoder=None, tokenizer=None,
        text_encoder_2=None, tokenizer_2=None,
        text_encoder_3=None, tokenizer_3=None,
        scheduler=scheduler)

    # Load image for editing
    print(f"Loading input image: {args.input_image}")
    image = Image.open(args.input_image)
    image = fix_longer_edge(image, image_size=args.image_size)
    print(f"Resized image to: {image.width}x{image.height}")

    print(f"Editing image with prompt: '{args.prompt}'")
    
    # Prepare messages with image input
    messages = [[{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": txt}]}]
                for txt in [args.prompt, args.negative_prompt]]

    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

    min_pixels = max_pixels = int(image.height * 28 / 32 * image.width * 28 / 32)
    inputs = processor(
        text=texts, images=[image]*2,
        min_pixels=min_pixels, max_pixels=max_pixels,
        videos=None, padding=True, return_tensors="pt").to("cuda")

    # Process with vision understanding
    input_ids, attention_mask, pixel_values, image_grid_thw = \
        inputs.input_ids, inputs.attention_mask, inputs.pixel_values, inputs.image_grid_thw

    input_ids = torch.cat([input_ids, input_ids.new_zeros(2, conditioner.config.num_queries)], dim=1)
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones(2, conditioner.config.num_queries)], dim=1)
    inputs_embeds = lmm.get_input_embeddings()(input_ids)
    inputs_embeds[:, -conditioner.config.num_queries:] = conditioner.meta_queries[None].expand(2, -1, -1)

    image_embeds = lmm.visual(pixel_values, grid_thw=image_grid_thw)
    image_token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    inputs_embeds[input_ids == image_token_id] = image_embeds

    lmm.model.rope_deltas = None
    outputs = lmm.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                        image_grid_thw=image_grid_thw, use_cache=False)

    hidden_states = outputs.last_hidden_state[:, -conditioner.config.num_queries:]
    prompt_embeds, pooled_prompt_embeds = conditioner(hidden_states)

    # Generate edited image
    edited_image = pipeline(
        image=image,
        prompt_embeds=prompt_embeds[:1],
        pooled_prompt_embeds=pooled_prompt_embeds[:1],
        negative_prompt_embeds=prompt_embeds[1:],
        negative_pooled_prompt_embeds=pooled_prompt_embeds[1:],
        height=image.height, width=image.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device=transformer.device).manual_seed(args.seed)
    ).images[0]

    edited_image.save(args.output)
    print(f"Edited image saved to: {args.output}")


if __name__ == "__main__":
    main()